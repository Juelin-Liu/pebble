from minibatch_model import GAT, SAGE, NeighborSampler
from minibatch_model import evaluate, test
from util import Config, Dataset, Timer
from util import get_args, get_load_compute_cores, get_num_numa, get_minibatch_meta, get_train_meta, load_dataset
from typing import List
import torch.multiprocessing as mp
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
    
def train(numa_id: int,
          config: Config, 
          data: Dataset,
          model: GAT | SAGE) -> Logger:

    print("start train", flush=True)

    loader_cores, compute_cores = get_load_compute_cores(numa_id=numa_id)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sampler = NeighborSampler(config.fanouts)
    dataloader = dgl.dataloading.DataLoader(data.graph, data.train_mask, sampler, device="cpu", batch_size=config.batch_size, shuffle=True, use_ddp=False, num_workers=len(loader_cores))
    
    timer = Timer()
    logger = Logger()    
    print("start loop", flush=True)

    with dataloader.enable_cpu_affinity(loader_cores=loader_cores, compute_cores=compute_cores, verbose=True):
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
                
            eval_acc = evaluate(config, data, model) if config.eval and numa_id == 0 else 0.0
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
            
            
    if numa_id == 0:
        print("done training")
        test_acc = test(config, data, model)
        log_minibatch_train(config, data, logger, test_acc)
        print("Test Accuracy {:.4f}".format(test_acc))
        
    return logger

def log_minibatch_train(config: Config, data: Dataset, log: Logger, test_acc: float):
    assert(config.log_file.endswith(".json"))
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
    data = load_dataset(config)
    model = None
    if config.model == "sage":
        model = SAGE(in_feats=data.in_feats, hid_feats=config.hid_size, num_layers=config.num_layers, out_feats=data.num_classes)
    elif config.model == "gat":
        model = GAT(in_feats=data.in_feats, hid_feats=config.hid_size, num_layers=config.num_layers, out_feats=data.num_classes, num_heads=config.num_head)
    else:
        print("unsupported model type", config.model)
        exit(-1)
        
    print("start training")

    model.share_memory()
    num_numa = get_num_numa()
    processes: List[mp.Process] = []

    print(mp.get_start_method())
    mp.set_start_method('spawn', force=True)
    for rank in range(num_numa):
        p = mp.Process(target=train, args=(rank, config, data, model))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    
if __name__ == "__main__":
    main()
    