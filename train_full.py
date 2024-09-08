import torch
import dataclasses
import json
import gc
from typing import List
from full_model import GAT, SAGE, GCN, eval_test
from util import *


def log_full_train(config: Config, data: Dataset, log: Logger):
    assert(config.log_file.endswith(".json"))
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_full_meta(config, data))
        ret.update(get_train_meta(config))
        ret.update(log.get_summary())
        ret["results"] = log.list()
        json.dump(ret, outfile, indent=4)
        
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
        eval_acc, test_acc = eval_test(data, model) if config.eval else 0.0
        evaluate_time = timer.record()
        cur_epoch_time = timer.stop() - evaluate_time
        acc_epoch_time += cur_epoch_time

        # logging
        log_epoch = LogEpoch(
            epoch=epoch,
            eval_acc=eval_acc,
            test_acc=test_acc,
            sample_time=0,
            load_time=0,
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
    log_full_train(config, data, logger)

if __name__ == "__main__":
    main()
