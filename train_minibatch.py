from minibatch_model import GAT, SAGE, GCN, NeighborSampler
from minibatch_model import evaluate, test
from util import Config, Dataset, Timer
from util import (
    get_args,
    get_load_compute_cores,
    get_minibatch_meta,
    get_train_meta,
    load_dataset,
)
from typing import List

import torch
import dgl
import dataclasses
import json


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


def train(data: Dataset, model: GAT | SAGE) -> Logger:
    config = model.config
    loader_cores, compute_cores = get_load_compute_cores()
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    sampler = NeighborSampler(config.fanouts)
    dataloader = dgl.dataloading.DataLoader(
        data.graph,
        data.train_mask,
        sampler,
        device="cpu",
        batch_size=config.batch_size,
        shuffle=True,
        use_ddp=False,
        num_workers=len(loader_cores),
    )

    timer = Timer()
    logger = Logger()
    model.train()

    with dataloader.enable_cpu_affinity(
        loader_cores=loader_cores, compute_cores=compute_cores, verbose=True
    ):
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
    return logger


def log_minibatch_train(config: Config, data: Dataset, log: Logger, test_acc: float):
    assert config.log_file.endswith(".json")
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_minibatch_meta(config, data))
        ret.update(get_train_meta(config))
        ret["test_acc"] = test_acc
        ret["results"] = log.list()
        json.dump(ret, outfile, indent=4)


def main():
    config = get_args()
    print(f"{config=}")
    data = load_dataset(config)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("creating graph formats")
    data.graph.create_formats_()
    gc.collect()
    torch.cuda.empty_cache()
    
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

    logger = train(data, model)    
    test_acc = test(data, model)
    log_minibatch_train(config, data, logger, test_acc)
    print("Test Accuracy {:.4f}".format(test_acc))


if __name__ == "__main__":
    main()
