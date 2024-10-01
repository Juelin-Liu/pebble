import torch
import dgl
import dataclasses
import json
import os
import gc
import quiver

from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from util import *
from typing import List, Union
from minibatch_model import GAT, SAGE, GCN
from minibatch_util import evaluate, test
from torch.multiprocessing import spawn

def ddp_setup(local_rank: int, config: Config, backend="nccl") -> DDPMeta:
    # assume torchrun creates one process at each host
    # each process than fork n processes where n equals number of gpus on a single host

    ddp_meta = DDPMeta()
    assert(ddp_meta.world_size == config.num_host) 

    ddp_meta.local_rank = local_rank
    ddp_meta.local_world_size = config.num_gpu_per_host
    ddp_meta.rank = ddp_meta.local_rank + ddp_meta.group_rank * ddp_meta.local_world_size
    ddp_meta.world_size = ddp_meta.local_world_size * ddp_meta.world_size
    ddp_meta.update()    
    dist.init_process_group(backend=backend, rank=ddp_meta.rank, world_size=ddp_meta.world_size)
    torch.cuda.set_device(local_rank)
    return ddp_meta

def ddp_exit():
    dist.destroy_process_group()

def get_model_ddp(config: Config, ddp_meta: DDPMeta, data: Dataset):
    model = None
    if config.model == "sage":
        model = SAGE(
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
            dropout=config.dropout
        )
    elif config.model == "gcn":
        model = GCN(
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
            dropout=config.dropout
        )
    elif config.model == "gat":
        model = GAT(
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
            num_heads=config.num_head,
            dropout=config.dropout
        )
    else:
        print("unsupported model type", config.model)
        exit(-1)
    
    device = torch.cuda.current_device()
    model = model.to(device)
    if ddp_meta.world_size > 1:
        model = DDP(model, device_ids=[ddp_meta.local_rank], output_device=ddp_meta.local_rank, find_unused_parameters=find_unused_parameters)
    return model

def get_quiver_p2p(config: Config, data: Dataset):
    assert(torch.cuda.is_available())
    assert(check_has_nvlink())

    cache_policy = "p2p_clique_replicate"
    device_cache_size = tensor_to_bytes(data.feat) // config.num_gpu_per_host
    
    device_list=[i for i in range(config.num_gpu_per_host)]
    # reserve 8GB for sampled subgraph etc
    subgraph_size = 8 * 1024 * 1024 * 1024

    # reserve space for caching graph topology data
    graph_size = (data.graph.num_edges() + data.graph.num_nodes()) * 8
    
    max_cache_memory = torch.cuda.get_device_properties(0).total_memory - graph_size - subgraph_size
    device_cache_size = min(device_cache_size, max_cache_memory)
    device_cache_size = max(0, device_cache_size)

    qfeat: quiver.Feature = quiver.Feature(rank=0, device_list=device_list, device_cache_size=device_cache_size, cache_policy=cache_policy)
    qfeat.from_cpu_tensor(data.feat)
    data.feat = qfeat

def train_quiver_p2p(local_rank: int, config: Config, packed):
    ddp_meta = ddp_setup(local_rank, config)
    device = torch.cuda.current_device()
    data = Dataset.unpack(packed)
    data.to(device)
    data.graph = data.graph.to(device)
    gc.collect()    

    print(f"start train {ddp_meta=} {device=}", flush=True)

    model = get_model_ddp(config, ddp_meta, data)
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    dataloader = dgl.dataloading.DataLoader(
        data.graph,
        data.train_mask,
        sampler,
        device=device,
        batch_size=config.batch_size,
        use_uva=config.sample_mode == "uva",
        use_ddp=ddp_meta.world_size > 1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    
    timer = Timer()
    logger = Logger()
    model.train()

    eval_acc = 0
    acc_epoch_time = 0.0

    dist.barrier()
    for epoch in range(config.num_epoch):
        
        timer.start()
        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        epoch_loss = 0
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            # done with sampling
            sample_time += timer.record()

            # feature data loading
            x = data.feat[input_nodes.type(torch.long)]
            tlabel = data.label[output_nodes]
            load_time += timer.record()

            # forward
            pred = model(blocks, x)
            loss = loss_fcn(pred, tlabel)
            forward_time += timer.record()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time += timer.record()
            epoch_loss += loss

        eval_acc = evaluate(config, data, model) if config.eval else 0.0
        test_acc = test(config, data, model) if config.eval else 0.0
        evaluate_time = timer.record()
        cur_epoch_time = timer.stop() - evaluate_time
        acc_epoch_time += cur_epoch_time

        # logging
        epoch_loss /= step
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss / (config.num_gpu_per_host * config.num_host)
        log_epoch = LogEpoch(
            epoch=epoch,
            eval_acc=eval_acc,
            test_acc=test_acc,
            sample_time=sample_time,
            load_time=load_time,
            forward_time=forward_time,
            backward_time=backward_time,
            cur_epoch_time=cur_epoch_time,
            acc_epoch_time=acc_epoch_time,
            evaluate_time=evaluate_time,
            loss=epoch_loss.item(),
        )

        if ddp_meta.local_rank == 0:
            log_epoch.print()
            logger.append(log_epoch)
    
    if ddp_meta.rank == 0:
        log_dgl_train(config, data, logger)

    ddp_exit()

def log_dgl_train(config: Config, data: Dataset, log: Logger):
    assert config.log_file.endswith(".json")
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_quiver_meta(config, data, check_has_nvlink()))
        ret.update(get_train_meta(config))
        ret.update(log.get_summary())
        ret["results"] = log.list()
        json.dump(ret, outfile, indent=4)
        print("log saved to", config.log_file, flush=True)


def main():
    if not check_has_nvlink():
        print("Does not have nvlink, cannot enable fast p2p transfer")
        exit(-1)
        
    config = get_args()
    start = time.time()
    data = load_dataset(config)
    end = time.time()
    get_quiver_p2p(config, data)
    print(f"loaded data in {round(end - start, 1)} secs", flush=True)
    gc.collect()
    packed = data.pack()
    device_list = [i for i in range(config.num_gpu_per_host)]
    quiver.init_p2p(device_list)
    try:
        spawn(train_quiver_p2p, args=(config, packed), nprocs=config.num_gpu_per_host, join=True)
    except Exception as e:
        print(f"error encountered with {config=}:", e)

if __name__ == "__main__":
    main()
