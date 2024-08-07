import dataclasses
import time
import dgl
import torch
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
import multiprocessing
from numa import numa_info
from typing import List

def get_num_threads():
    return multiprocessing.cpu_count()

def get_num_cores():
    return torch.get_num_threads()

def get_list_cores(numa_id: int = 0):
    return numa_info[numa_id]

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        if self.start_time is not None:
            raise RuntimeError("Timer is already running. Use stop() to stop it before starting again.")
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not running. Use start() to start it.")
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.start_time = None  # Reset the timer
        self.end_time = None
        return elapsed_time

@dataclasses.dataclass
class Config:
    batch_size:int # Minibatch only
    fanouts: List[int] # Minibatch only
    num_epoch: int
    hid_size: int
    num_head: int
    world_size: int
    num_partition: int
    graph_name: str
    data_dir: str
    model: str
    log_file: str
    eval: bool
    
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.fanouts = args.fanouts
        self.num_epoch = args.num_epoch
        self.hid_size = args.hid_size
        self.num_layers = args.num_layers
        self.num_head = args.num_head
        self.world_size = args.world_size
        self.num_partition = args.num_partition
        self.graph_name = args.graph_name
        self.data_dir = args.data_dir
        self.model = args.model
        self.log_file = args.log_file
        self.eval = args.eval
        
@dataclasses.dataclass
class Dataset:
    graph: dgl.DGLGraph
    feat: torch.Tensor
    label: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_classes: int
    in_feats: int
    
    def __init__(self, graph, feat, label, train_mask, val_mask, test_mask, num_classes, in_feats):
        self.graph = graph
        self.feat = feat
        self.label = label
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = num_classes
        self.in_feats = in_feats
        

def get_args()->Config:
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--batch_size', default=1024, type=int, help='Global batch size (default: 1024)')
    parser.add_argument('--fanouts', default="15,15,15", type=lambda fanouts : [int(fanout) for fanout in fanouts.split(',')], help='Fanouts')
    parser.add_argument('--num_epoch', default=1, type=int, help='Number of epochs to train (default 1)')
    parser.add_argument('--hid_size', default=256, type=int, help="Model hidden dimension")
    parser.add_argument('--num_layers', default=3, type=int, help="Model layers")
    parser.add_argument('--num_head', default=4, type=int, help="GAT only: number of attention head")
    parser.add_argument('--world_size', default=1, type=int, help='Number of Hosts')
    parser.add_argument('--num_partition', default=1, type=int, help='Number of partitions')
    parser.add_argument('--graph_name', default="ogbn-arxiv", type=str, help="Input graph name", choices=["ogbn-proteins", "ogbn-products", "ogbn-arxiv", "ogbn-mag", "ogbn-papers100M"])
    # parser.add_argument('--data_dir', required=True, type=str, help="Root data directory")
    parser.add_argument('--data_dir', default="/data/juelin/dataset/gnn", type=str, help="Root data directory")
    parser.add_argument('--model', default="gat", type=str, help="Model type", choices=["gcn", "gat", "sage"])
    parser.add_argument('--log_file',default='log.csv',type=str,help='output log file')
    parser.add_argument('--eval',default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    config = Config(args)
    return config

def load_dataset(config: Config):
    if "ogbn" in config.graph_name:
        dataset = DglNodePropPredDataset(name=config.graph_name, root=config.data_dir)
        g, label = dataset[0]
        g = dgl.add_self_loop(g)
        label = torch.flatten(label).to(torch.int64)
        feat = g.ndata.pop("feat")
        in_feats = feat.shape[1]
        num_classes = dataset.num_classes
        idx_split = dataset.get_idx_split()
        train_mask = idx_split["train"]
        val_mask = idx_split["valid"]
        test_mask = idx_split["test"]
        return Dataset(graph=g, feat=feat, label=label, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, num_classes=num_classes, in_feats=in_feats)
        