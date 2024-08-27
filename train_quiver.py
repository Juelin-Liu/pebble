from minibatch_model import GAT, SAGE, GCN, NeighborSampler
from minibatch_model import evaluate, test
from util import *
from typing import List
from typing import Union

import torch
import dgl
import dataclasses
import json
import quiver

@dataclasses.dataclass
class LogEpoch:
    epoch: int
    eval_acc: float
    sample_time: float
    load_time: float
    forward_time: float
    backward_time: float
    cur_epoch_time: float  # exclude evaluate time
    acc_epoch_time: float  # accumulative epoch time excluding evaluate time
    evaluate_time: float
    loss: float

    def print(self):
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Epoch Time {:.4f}".format(
                self.epoch, self.loss, self.eval_acc, self.cur_epoch_time
            ),
            flush=True,
        )

    def dict(self):
        return self.__dict__


@dataclasses.dataclass
class Logger:
    steps: List[LogEpoch] = None

    def __init__(self):
        self.steps = []

    def append(self, step: LogEpoch):
        self.steps.append(step)

    def list(self):
        ret = []
        for step in self.steps:
            ret.append(step.dict())

        return ret

@dataclasses.dataclass
class DDPMeta:
    local_rank : int
    group_rank : int
    rank: int
    local_world_size : int
    world_size: int

    def __init__(self):
        local_rank = int(os.environ["LOCAL_RANK"])
        group_rank = int(os.environ["GROUP_RANK"])
        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        world_size = int(os.environ["WORLD_SIZE"])
    
def ddp_setup(backend="nccl"):
    torch.cuda.set_device(metadata.local_rank)
    dist.init_process_group(backend=backend)
    return metadata

def ddp_exit():
    dist.destroy_process_group()

def load_data(config: Config):
    # TODO: load dataset only in local_rank 0 and shared it via shared memory
    return load_dataset(config)

def train_ddp(data: Dataset, model: Union[GAT, SAGE, GCN], feat: quiver.Feature):
    metadata = ddp_setup()
    device = torch.cuda.current_device()
    num_workers = 0

    if config.sample_mode == "uva":
        graph = data.graph.pin_memory()
    elif config.sample_mode == "gpu":
        graph = graph.to(device)

    sampler = NeighborSampler(config.fanouts)
    dataloader = DataLoader(
        graph,
        torch.arange(g.num_nodes()),
        sampler,
        device=device,
        batch_size=batch_size,
        use_uva=config.sample_mode == "uva",
        use_ddp=config.world_size > 1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

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
        log_epoch = LogEpoch(
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

        log_epoch.print()
        logger.append(log_epoch)

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
    print(f"{config=}")


    data = load_data(config)
    model = None
    if config.model == "sage":
        model = SAGE(
            config=config,
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
        )
    elif config.model == "gcn":
        model = GCN(
            config=config,
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
        )
    elif config.model == "gat":
        model = GAT(
            config=config,
            in_feats=data.in_feats,
            hid_feats=config.hid_size,
            num_layers=config.num_layers,
            out_feats=data.num_classes,
            num_heads=config.num_head,
        )
    else:
        print("unsupported model type", config.model)
        exit(-1)

    rcfg = DDPMeta()
    device_list = [i for i in range(rcfg.local_world_size)]
    
    indptr, indices, _ = data.graph.adj_tensors("csc")
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    has_nvlink = "A100" in get_cuda_gpu_model()
    cache_policy = "p2p_clique_replicate" if has_nvlink else "device_replicate"
    device_cache_size = str(int(tensor_to_bytes(data.feat) / (1024 * 1024))) + "MB"

    if has_nvlink:
        device_cache_size = str(int(tensor_to_bytes(data.feat) / (1024 * 1024 * num_gpus))) + "MB"

    qfeat = quiver.Feature(rank=0, device_list=device_list, device_cache_size=device_cache_size, cache_policy=cache_policy)

    logger = train(data, model, qfeat)
    test_acc = test(data, model, qfeat)
    log_minibatch_train(config, data, logger, test_acc)
    print("Test Accuracy {:.4f}".format(test_acc))


if __name__ == "__main__":
    main()