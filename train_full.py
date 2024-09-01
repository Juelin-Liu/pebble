import torch
import dataclasses
import json
import gc

from typing import List
from full_model import GAT, SAGE, GCN, evaluate, test
from util import Timer, Config, Dataset, get_args, load_dataset, get_full_meta, get_train_meta


@dataclasses.dataclass
class LogEpoch:
    epoch: int
    eval_acc: float
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
    
def log_full_train(config: Config, data: Dataset, log: Logger, test_acc: float):
    assert(config.log_file.endswith(".json"))
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_full_meta(config, data))
        ret.update(get_train_meta(config))
        ret["test_acc"] = test_acc
        ret["results"] = log.list()
        json.dump(ret, outfile)
        
def train(data: Dataset, model: GAT | GCN) -> Logger:
    config = model.config
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    logger = Logger()
    timer = Timer()
    acc_epoch_time = 0.0
    # training loop
    for epoch in range(config.num_epoch):
        timer.start()

        # forward
        model.train()
        logits = model(data.graph, data.feat)
        pred = logits[data.train_mask]
        tlabel = data.label[data.train_mask]
        loss: torch.Tensor = loss_fcn(pred, tlabel)
        forward_time = timer.record()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = timer.record()

        # evaluation
        eval_acc = evaluate(data, model) if config.eval else 0.0
        evaluate_time = timer.record()
        cur_epoch_time = timer.stop() - evaluate_time
        acc_epoch_time += cur_epoch_time

        # logging
        log_epoch = LogEpoch(
            epoch=epoch,
            eval_acc=eval_acc,
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

def main():
    config = get_args()
    data = load_dataset(config)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("creating graph formats")
    data.graph.create_formats_()
    
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("creating models")

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
    log_full_train(config, data, logger, test_acc)
    print("Test Accuracy {:.4f}".format(test_acc))


if __name__ == "__main__":
    main()
