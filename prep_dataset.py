import torch
import dgl
import dataclasses
import json
import os
import quiver
import gc

from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from util import *
from typing import List, Union
from minibatch_model import GAT, SAGE, GCN
from minibatch_util import evaluate, test
from torch.multiprocessing import spawn
import numpy as np

def main():
    config = get_args()
    print(f"{config=}", flush=True)

    data: Dataset = load_dataset(config)    

    out_dir = os.path.join(config.data_dir, config.graph_name)
    os.makedirs(out_dir, exist_ok=True)
    print("creating graph formats", flush=True)
    np.save(f"{out_dir}/feat.npy", data.feat.detach().cpu().numpy())
    del data.feat
    gc.collect()

    csc = data.graph.formats("csc").int()
    indptr, indices, _ = csc.adj_tensors("csc")
    np.save(f"{out_dir}/indptr.npy", indptr.detach().cpu().numpy())
    np.save(f"{out_dir}/indices.npy", indices.detach().cpu().numpy())
    np.save(f"{out_dir}/train_idx.npy", data.train_mask.int().detach().cpu().numpy())
    np.save(f"{out_dir}/test_idx.npy", data.test_mask.int().detach().cpu().numpy())
    np.save(f"{out_dir}/valid_idx.npy", data.val_mask.int().detach().cpu().numpy())
    np.save(f"{out_dir}/label.npy", data.label.detach().cpu().numpy())

if __name__ == "__main__":
    main()
