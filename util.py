import dataclasses
import time
import dgl
import torch
import argparse
import subprocess
import os

from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from numa import numa_info
from typing import List

find_unused_parameters=True

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
    
@dataclasses.dataclass
class LogEpoch:
    epoch: int
    eval_acc: float
    test_acc: float
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
            "Epoch {:05d} | Loss {:.4f} | Evaluation Accuracy {:.4f} | Test Accuracy {:.4f} | Epoch Time {:.4f}".format(
                self.epoch, self.loss, self.eval_acc, self.test_acc, self.cur_epoch_time
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

    def get_summary(self):
        best_eval_acc = 0.0
        best_eval_idx = 0
        time_to_best_eval = 0
        
        best_test_acc = 0.0
        best_test_idx = 0
        time_to_best_test = 0
        
        for idx, step in enumerate(self.steps):
            if step.eval_acc > best_eval_acc:
                best_eval_acc = step.eval_acc
                best_eval_idx = idx
                time_to_best_eval = step.acc_epoch_time
                
            if step.test_acc > best_test_acc:
                best_test_acc = step.test_acc
                best_test_idx = idx
                time_to_best_test = step.acc_epoch_time
                
        meta = {}
        meta["best_eval_acc"] = best_eval_acc
        meta["best_eval_epoch"] = best_eval_idx + 1
        meta["time_to_best_eval"] = time_to_best_eval
        
        meta["best_test_acc"] = best_test_acc
        meta["best_test_epoch"] = best_test_idx + 1
        meta["time_to_best_test"] = time_to_best_test
        return meta
    

class Timer:
    def __init__(self):
        self.start_time = None
        self.last_record = None
        self.end_time = None

    def start(self):
        if self.start_time is not None:
            raise RuntimeError(
                "Timer is already running. Use stop() to stop it before starting again."
            )
        self.start_time = time.time()
        self.last_record = time.time()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not running. Use start() to start it.")
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.start_time = None  # Reset the timer
        self.end_time = None
        self.last_record = None
        return elapsed_time

    def record(self):
        if self.last_record is None:
            raise RuntimeError("Timer is not running. Use start() to start it.")
        current = time.time()
        elapsed_time = current - self.last_record
        self.last_record = time.time()  # Reset the timer
        return elapsed_time


gconfig = None

@dataclasses.dataclass
class Config:
    sample_mode: str
    batch_size: int  # Minibatch only
    fanouts: List[int]  # Minibatch only
    num_epoch: int
    hid_size: int
    num_head: int
    lr: float
    weight_decay: float
    dropout: float
    num_host: int
    num_gpu_per_host: int
    graph_name: str
    data_dir: str
    model: str
    log_file: str
    eval: bool

    def __init__(self, args):
        self.sample_mode = args.sample_mode
        self.batch_size = args.batch_size
        self.fanouts = args.fanouts
        self.num_epoch = args.num_epoch
        self.hid_size = args.hid_size
        self.num_layers = len(self.fanouts)
        self.num_head = args.num_head
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_host = args.num_host
        self.dropout = args.dropout
        self.num_gpu_per_host = args.num_gpu_per_host
        self.graph_name = args.graph_name
        self.data_dir = args.data_dir
        self.model = args.model
        self.log_file = args.log_file
        self.eval = args.eval

    @staticmethod
    def get_global_config():
        global gconfig
        return gconfig
    
    @staticmethod
    def set_global_config(config):
        global gconfig
        gconfig = config
        
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

    def __init__(
        self, graph, feat, label, train_mask, val_mask, test_mask, num_classes, in_feats
    ):
        self.graph = graph.int()
        self.feat = feat
        self.label = label.type(torch.long)
        self.train_mask = train_mask.type(torch.int)
        self.val_mask = val_mask.type(torch.int)
        self.test_mask = test_mask.type(torch.int)
        self.num_classes = num_classes
        self.in_feats = in_feats

    def to(self, device):
        self.label = self.label.to(device)
        self.train_mask = self.train_mask.type(torch.int).to(device)
        self.val_mask = self.val_mask.type(torch.int).to(device)
        self.test_mask = self.test_mask.type(torch.int).to(device)

    def pack(self):
        return (self.graph, self.feat, self.label, self.train_mask, self.val_mask, self.test_mask, self.num_classes, self.in_feats)
    
    @classmethod
    def unpack(cls, packed):
        graph, feat, label, train_mask, val_mask, test_mask, num_classes, in_feats = packed
        return cls(graph, feat, label, train_mask, val_mask, test_mask, num_classes, in_feats)
    
def get_num_numa():
    return len(numa_info.keys())

def get_load_compute_cores(numa_id: int = 0):
    all_threads = numa_info[numa_id]
    num_cores = len(all_threads) // 2
    loader_cores = all_threads[:num_cores]
    compute_cores = all_threads[num_cores:]
    return loader_cores, compute_cores

def str_to_bytes(cache_size: str):
    if "G" in cache_size:
        n, _ = cache_size.split('G')
        return float(n) * 1024 * 1024 * 1024
    elif 'M' in cache_size:
        n, _ = cache_size.split('M')
        return float(n) * 1024 * 1024
    elif 'K' in cache_size:
        n, _ = cache_size.split('K')
        return float(n) * 1024

def tensor_to_bytes(t: torch.Tensor) -> int:
    sz = t.shape[0] * t.shape[1] * 4
    return sz

def check_has_nvlink()->bool:
    result = subprocess.run(['nvidia-smi', 'nvlink', '-s'], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if (result.stdout == ""):
        return False
    return True

def get_cuda_gpu_model():
    try:
        # Execute the nvidia-smi command
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if there was an error
        if result.returncode != 0:
            print("Error executing nvidia-smi:", result.stderr)
            return []
        
        # Parse the output into a list of GPU model
        gpu_model = result.stdout.strip().split('\n')
        
        return gpu_model

    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed and accessible.")
        return []

def get_cpu_model() -> str:
    ret = subprocess.check_output("lscpu", shell=True).strip().decode()
    idx = ret.find("Model name:")
    cpu_model = ret[idx+11:].strip().split("\n")[0]
    return cpu_model

def get_train_meta(config: Config):
    ret = dict()
    ret["weight_decay"] = config.weight_decay
    ret["learning_rate"] = config.lr
    ret["dropout"] = config.dropout
    return ret

def get_full_meta(config: Config, data: Dataset):
    ret = dict()
    ret["graph_name"] = config.graph_name
    ret["train_mode"] = "full"
    ret["cpu_model"] = get_cpu_model()
    ret["num_node"] = data.graph.num_nodes()
    ret["num_edge"] = data.graph.num_edges()
    ret["feat_width"] = data.in_feats
    ret["num_epoch"] = config.num_epoch
    ret["num_partition"] = config.num_gpu_per_host
    return ret

def get_minibatch_meta(config: Config, data: Dataset):
    ret = dict()
    ret["graph_name"] = config.graph_name
    ret["train_mode"] = "minibatch"
    ret["num_node"] = data.graph.num_nodes()
    ret["num_edge"] = data.graph.num_edges()
    ret["cpu_model"] = get_cpu_model()
    ret["feat_width"] = data.in_feats
    ret["batch_size"] = config.batch_size
    ret["fanouts"] = config.fanouts
    ret["num_epoch"] = config.num_epoch
    ret["num_partition"] = config.num_gpu_per_host
    return ret

def get_quiver_meta(config: Config, data: Dataset, use_p2p = False):
    ret = dict()
    ret["system_name"] = "quiver-p2p" if use_p2p else "quiver-loc"
    ret["train_mode"] = "minibatch"
    ret["graph_name"] = config.graph_name
    ret["sample_mode"] = config.sample_mode
    ret["num_node"] = data.graph.num_nodes()
    ret["num_edge"] = data.graph.num_edges()
    ret["cpu_model"] = get_cpu_model()
    ret["gpu_model"] = get_cuda_gpu_model()
    ret["feat_width"] = data.in_feats
    ret["batch_size"] = config.batch_size
    ret["fanouts"] = config.fanouts
    ret["num_epoch"] = config.num_epoch
    ret["num_gpu_per_host"] = config.num_gpu_per_host
    ret["num_host"] = config.num_host
    
    return ret

def get_dgl_meta(config: Config, data: Dataset):
    ret = dict()
    ret["system_name"] = "dgl"
    ret["train_mode"] = "minibatch"
    ret["graph_name"] = config.graph_name
    ret["sample_mode"] = config.sample_mode
    ret["num_node"] = data.graph.num_nodes()
    ret["num_edge"] = data.graph.num_edges()
    ret["cpu_model"] = get_cpu_model()
    ret["gpu_model"] = get_cuda_gpu_model()
    ret["feat_width"] = data.in_feats
    ret["batch_size"] = config.batch_size
    ret["fanouts"] = config.fanouts
    ret["num_epoch"] = config.num_epoch
    ret["num_gpu_per_host"] = config.num_gpu_per_host
    ret["num_host"] = config.num_host
    return ret

def get_args() -> Config:
    parser = argparse.ArgumentParser(description="local run script")
    parser.add_argument(
        "--sample_mode", default="gpu", type=str, help="Sample device (default: gpu)", choices=["gpu", "uva", "cpu"]
    )
    
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Global batch size (default: 1024)"
    )
    
    parser.add_argument(
        "--fanouts",
        default="15,15,15",
        type=lambda fanouts: [int(fanout) for fanout in fanouts.split(",")],
        help="Fanouts",
    )
    
    parser.add_argument(
        "--num_epoch", default=1, type=int, help="Number of epochs to train (default 1)"
    )
    parser.add_argument(
        "--hid_size", default=256, type=int, help="Model hidden dimension"
    )

    parser.add_argument(
        "--num_head", default=4, type=int, help="GAT only: number of attention head"
    )
    
    parser.add_argument(
        "--lr", default=5e-3, type=float, help="learning rate"
    )
    
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay"
    )
    
    parser.add_argument(
        "--dropout", default=0.5, type=float, help="dropout ratio"
    )
    
    parser.add_argument("--num_host", default=1, type=int, help="Number of Hosts")
    parser.add_argument(
        "--num_gpu_per_host", default=1, type=int, help="Number of gpus on a single host"
    )
    
    parser.add_argument(
        "--graph_name",
        default="ogbn-arxiv",
        type=str,
        help="Input graph name",
        choices=[
            "ogbn-proteins",
            "pubmed",
            "reddit",
            "orkut",
            "ogbn-products",
            "ogbn-arxiv",
            "ogbn-mag",
            "ogbn-papers100M",
        ],
    )
    
    parser.add_argument('--data_dir', required=True, type=str, help="Root data directory")
    # parser.add_argument(
    #     "--data_dir",
    #     default="/data/juelin/dataset/gnn",
    #     type=str,
    #     help="Root data directory",
    # )
    
    parser.add_argument(
        "--model",
        default="gat",
        type=str,
        help="Model type",
        choices=["gcn", "gat", "sage"],
    )
    parser.add_argument(
        "--log_file", default="log.json", type=str, help="output log file"
    )

    parser.add_argument("--eval", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    config = Config(args)
    Config.set_global_config(config)
    return config

def load_dataset(config: Config, topo_only=False):
    if config.graph_name == "ogbn-papers100M" or config.graph_name == "orkut":
        import numpy as np
        in_dir = os.path.join(config.data_dir, config.graph_name)
        indptr = np.load(f"{in_dir}/indptr.npy")
        indices = np.load(f"{in_dir}/indices.npy")
        weights = np.empty([])
        g = dgl.graph(data=("csc", (indptr, indices, weights)))
        feat = np.load(f"{in_dir}/feat.npy")
        feat = torch.from_numpy(feat)
        train_mask = np.load(f"{in_dir}/train_idx.npy")
        train_mask = torch.from_numpy(train_mask)
        val_mask = np.load(f"{in_dir}/valid_idx.npy")
        val_mask = torch.from_numpy(val_mask)
        test_mask = np.load(f"{in_dir}/test_idx.npy")
        test_mask = torch.from_numpy(test_mask)
        label = np.load(f"{in_dir}/label.npy")
        label = torch.from_numpy(label)
        in_feats = feat.shape[1]
        num_classes = torch.max(label).item() + 1
        
        data = Dataset(
            graph=g,
            feat=feat,
            label=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=num_classes,
            in_feats=in_feats,
        )
        return data
    
    elif "ogbn" in config.graph_name:
        dataset = DglNodePropPredDataset(name=config.graph_name, root=config.data_dir)
        g, label = dataset[0]
        g = dgl.add_self_loop(g)
        label = torch.flatten(label).to(torch.int64)
        feat = g.ndata.pop("feat")
        idx_split = dataset.get_idx_split()
        train_mask = idx_split["train"]
        val_mask = idx_split["valid"]
        test_mask = idx_split["test"]
        in_feats = feat.shape[1]
        num_classes = dataset.num_classes

        if topo_only:
            feat = None

        return Dataset(
            graph=g,
            feat=feat,
            label=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=num_classes,
            in_feats=in_feats,
        )

    elif config.graph_name in ["pubmed", "reddit"]:
        dataset = None
        if config.graph_name == "pubmed":
            dataset = dgl.data.PubmedGraphDataset(
                raw_dir=config.data_dir, transform=dgl.add_self_loop
            )
        elif config.graph_name == "reddit":
            dataset = dgl.data.RedditDataset(
                raw_dir=config.data_dir, transform=dgl.add_self_loop
            )

        g: dgl.DGLGraph = dataset[0]
        indices = torch.arange(g.num_nodes())
        label = g.ndata.pop("label")
        feat = g.ndata.pop("feat")
        train_mask = indices[g.ndata.pop("train_mask")]
        val_mask = indices[g.ndata.pop("val_mask")]
        test_mask = indices[g.ndata.pop("test_mask")]

        label = torch.flatten(label).to(torch.int64)
        in_feats = feat.shape[1]
        num_classes = dataset.num_classes

        if topo_only:
            feat = None

        return Dataset(
            graph=g,
            feat=feat,
            label=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=num_classes,
            in_feats=in_feats,
        )
    
    elif config.graph_name == "orkut":

        dataset_dir = os.path.join(config.data_dir, "orkut")
        graph = dgl.load_graphs(os.path.join(dataset_dir, "orkut_bidirected.dgl"))
        graph = graph[0][0]
        node_features = graph.ndata.pop("feat")
        in_feats = node_features.shape[1]
        label = graph.ndata.pop("label")
        num_classes = (label.max() + 1).item()
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.arange(start=0, end=n_train)
        val_mask = torch.arange(start=n_train, end=n_train + n_val)
        test_mask = torch.arange(start=n_train + n_val, end=n_nodes)
        
        print("construct dataset", flush=True)
        return Dataset(
            graph=graph,
            feat=node_features,
            label=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=num_classes,
            in_feats=in_feats,
        )
