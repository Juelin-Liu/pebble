from minibatch_model import GAT, SAGE, GCN, NeighborSampler
from minibatch_model import eval_test
from util import *
from typing import List

import torch
import dgl
import dataclasses
import json
import gc
from tqdm import tqdm


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

        # for epoch in tqdm(range(config.num_epoch)):
        for epoch in range(config.num_epoch):
            timer.start()
            sample_time = 0
            load_time = 0
            forward_time = 0
            backward_time = 0

            # for step, (input_nodes, output_nodes, blocks) in enumerate(tqdm(dataloader, leave=False)):
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

            eval_acc, test_acc = eval_test(data, model) if config.eval else 0.0
            evaluate_time = timer.record()
            cur_epoch_time = timer.stop() - evaluate_time
            acc_epoch_time += cur_epoch_time

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

            log_epoch.print()
            logger.append(log_epoch)
    return logger


def log_minibatch_train(config: Config, data: Dataset, log: Logger):
    assert config.log_file.endswith(".json")
    with open(config.log_file, "w") as outfile:
        ret = dict()
        ret["version"] = 1
        ret.update(get_minibatch_meta(config, data))
        ret.update(get_train_meta(config))
        ret.update(log.get_summary())
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
    log_minibatch_train(config, data, logger)

if __name__ == "__main__":
    main()
