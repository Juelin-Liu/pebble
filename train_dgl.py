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
        
def ddp_setup(backend="nccl") -> DDPMeta:
    ddp_meta = DDPMeta()
    torch.cuda.set_device(ddp_meta.local_rank)
    dist.init_process_group(backend=backend)
    return ddp_meta

def ddp_exit():
    dist.destroy_process_group()

def load_data_ddp(config: Config, ddp_meta: DDPMeta):
    # TODO: load dataset only in local_rank 0 and shared it via shared memory
    data = load_dataset(config)
    data.to(torch.cuda.current_device())
    return data

def get_model_ddp(config: Config, ddp_meta: DDPMeta, data: Dataset):
    model = None
    if config.model == "sage":
        model = SAGE(
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
        )
    elif config.model == "gcn":
        model = GCN(
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
        )
    elif config.model == "gat":
        model = GAT(
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
            num_heads=config.num_head,
        )
    else:
        print("unsupported model type", config.model)
        exit(-1)
    
    device = torch.cuda.current_device()
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[ddp_meta.rank])
    return ddp_model

def train_dgl_ddp(config: Config):
    ddp_meta = ddp_setup()
    device = torch.cuda.current_device()
    data = load_data_ddp(config, ddp_meta)
    model = get_model_ddp(config, ddp_meta, data)
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)

    if config.sample_mode == "uva":
        data.graph.pin_memory_()
    elif config.sample_mode == "gpu":
        data.graph = data.graph.to(device)
    g = data.graph
    data.feat = data.feat.to(device)
    dataloader = dgl.dataloading.DataLoader(
        g,
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

        eval_acc = evaluate(data, model) if config.eval else 0.0
        evaluate_time = timer.record()
        cur_epoch_time = timer.stop() - evaluate_time
        acc_epoch_time += cur_epoch_time

        # logging
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
            loss=loss.item(),
        )

        if ddp_meta.rank == 0:
            log_step.print()
            logger.append(log_step)

def log_minibatch_train(config: Config, data: Dataset, log: Logger, test_acc: float):
    assert config.log_file.endswith(".json")
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_minibatch_meta(config, data))
        ret.update(get_train_meta(config))
        ret["test_acc"] = test_acc
        ret["results"] = log.list()
        json.dump(ret, outfile)


def main():
    config = get_args()
    train_dgl_ddp(config)

if __name__ == "__main__":
    main()