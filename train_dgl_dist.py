import torch
import dgl
import dataclasses
import json
import os
import gc

from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from util import *
from typing import List, Union
from minibatch_model import GAT, SAGE, GCN
from minibatch_util import evaluate, test
from torch.multiprocessing import spawn
    
def ddp_setup(config: Config, backend="nccl") -> DDPMeta:
    # assume torchrun creates one process at each host
    # each process than fork n processes where n equals number of gpus on a single host

    ddp_meta = DDPMeta()
    assert(ddp_meta.world_size == config.num_host * config.num_gpu_per_host)  
    torch.cuda.set_device(ddp_meta.local_rank)
    dist.init_process_group(backend=backend, rank=ddp_meta.rank, world_size=ddp_meta.world_size)
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

def train_dgl_ddp(ddp_meta: DDPMeta, config: Config, data: Dataset):
    device = torch.cuda.current_device()
    data.to(device)
    data.feat = data.feat.to(device)
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
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
        # logging
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
            loss=loss.item(),
        )

        if ddp_meta.rank == 0:
            log_epoch.print()
            logger.append(log_epoch)
    
    if ddp_meta.rank == 0:
        log_dgl_train(config, data, logger)

def log_dgl_train(config: Config, data: Dataset, log: Logger):
    assert config.log_file.endswith(".json")
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_dgl_meta(config, data))
        ret.update(get_train_meta(config))
        ret.update(log.get_summary())
        ret["results"] = log.list()
        json.dump(ret, outfile, indent=4)
        print("log saved to", config.log_file, flush=True)


def main():
    config = get_args()
    ddp_meta = ddp_setup(config)
    rank = ddp_meta.rank
    start = time.time()
    data = load_dataset(config)
    end = time.time()
    
    print(f"{rank=} loaded data in {round(end - start, 1)} secs", flush=True)
    gc.collect()
    train_dgl_ddp(ddp_meta, config, data)
    ddp_exit()

if __name__ == "__main__":
    main()
