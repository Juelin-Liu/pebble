import torch
import dgl
import dataclasses
import json
import os
import quiver
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from util import *
from typing import List, Union
from minibatch_model import GAT, SAGE, GCN
from minibatch_util import evaluate, test
from torch.multiprocessing import spawn

@dataclasses.dataclass
class DDPMeta:
    local_rank : int
    group_rank : int
    rank: int
    local_world_size : int
    world_size: int
    def __init__(self):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.group_rank = int(os.environ["GROUP_RANK"])
        self.rank = int(os.environ["RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.world_size = int(os.environ["WORLD_SIZE"])
    
    def update(self):
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["GROUP_RANK"] = str(self.group_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.local_world_size)
        os.environ["WORLD_SIZE"] = str(self.world_size)
    
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
    ddp_model = DDP(model, device_ids=[ddp_meta.local_rank])
    return ddp_model

def train_quiver_ddp(rank: int, config: Config, qfeat: quiver.Feature, packed):
    ddp_meta = ddp_setup(rank, config)
    device = torch.cuda.current_device()
    print(f"start train quiver {ddp_meta=} {device=}", flush=True)

    data = Dataset.unpack(packed)
    data.feat = qfeat

    model = get_model_ddp(config, ddp_meta, data)
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)

    if config.sample_mode == "uva":
        data.graph.pin_memory_()
    elif config.sample_mode == "gpu":
        data.graph = data.graph.formats("csc")
        data.graph = data.graph.to(device)
    
    data.to(device)
    dataloader = dgl.dataloading.DataLoader(
        data.graph,
        data.train_mask,
        sampler,
        device=device,
        batch_size=config.batch_size,
        use_uva=config.sample_mode == "uva",
        use_ddp=ddp_meta.world_size > 1,
        shuffle=True,
        drop_last=False,
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
            x = data.feat[input_nodes]
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
        evaluate_time = timer.record()
        cur_epoch_time = timer.stop() - evaluate_time
        acc_epoch_time += cur_epoch_time

        # logging
        epoch_loss /= step
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
        log_step = LogStep(
            epoch=epoch,
            eval_acc=eval_acc,
            sample_time=sample_time,
            load_time=load_time,
            forward_time=forward_time,
            backward_time=backward_time,
            cur_epoch_time=cur_epoch_time,
            acc_epoch_time=acc_epoch_time,
            evaluate_time=evaluate_time,
            loss=epoch_loss.item(),
        )

        if ddp_meta.rank == 0:
            log_step.print()
            logger.append(log_step)
    
    test_acc = test(config, data, model)
    if ddp_meta.rank == 0:
        log_quiver_train(config, data, logger, test_acc)

def get_quiver_feat(config: Config, data: Dataset):
    assert(torch.cuda.is_available())
    assert(config.num_gpu_per_host <= torch.cuda.device_count())
    
    device_list = [i for i in range(config.num_gpu_per_host)]
    indptr, indices, _ = data.graph.adj_tensors("csc")
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    
    gpu_model = get_cuda_gpu_model()
    has_nvlink = check_has_nvlink()
    cache_policy = "p2p_clique_replicate" if has_nvlink else "device_replicate"
    # device_cache_size = str(int(tensor_to_bytes(data.feat) / (1024 * 1024))) + "MB"
    device_cache_size = tensor_to_bytes(data.feat)

    if has_nvlink:
        quiver.init_p2p(device_list=device_list)
        device_cache_size = device_cache_size // config.num_gpu_per_host
    
    # reserve 8GB for sampled subgraph etc
    subgraph_size = 8 * 1024 * 1024 * 1024

    # reserve space for caching graph topology data
    graph_size = (data.graph.num_edges() * 2 + data.graph.num_nodes()) * 4
    
    max_cache_memory = torch.cuda.get_device_properties(0).total_memory - graph_size - subgraph_size
    device_cache_size = min(device_cache_size, max_cache_memory)
    device_cache_size = max(0, device_cache_size)

    qfeat: quiver.Feature = quiver.Feature(rank=torch.cuda.current_device(), device_list=device_list, device_cache_size=device_cache_size, cache_policy=cache_policy, csr_topo=csr_topo)
    qfeat.from_cpu_tensor(data.feat)
    return qfeat

def log_quiver_train(config: Config, data: Dataset, log: Logger, test_acc: float):
    assert config.log_file.endswith(".json")
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_quiver_meta(config, data))
        ret.update(get_train_meta(config))
        ret["test_acc"] = test_acc
        ret["results"] = log.list()
        json.dump(ret, outfile, indent=4)
        print("log saved to", config.log_file, flush=True)


def main():
    config = get_args()
    print(f"{config=}", flush=True)
    data = load_dataset(config)
    data.graph = data.graph.int()
    data.graph.create_formats_()
    qfeat = get_quiver_feat(config, data)
    packed = data.pack()
    
    try:
        spawn(train_quiver_ddp, args=(config, qfeat, packed), nprocs=config.num_gpu_per_host, join=True)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        
if __name__ == "__main__":
    main()