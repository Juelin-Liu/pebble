import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchmetrics.functional as MF
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)

from typing import Union
from minibatch_model import SAGE, GAT, GCN
from util import Dataset, Config, tensor_to_bytes

MODEL_TYPES = Union[SAGE, GAT, GCN]

def inference_full(config: Config, data: Dataset, model: DDP, mode):
    assert(mode in ["test", "eval"])
    device = torch.cuda.current_device()
    sampler = MultiLayerFullNeighborSampler(1)
    _model: MODEL_TYPES = model.module
    mask = data.test_mask if mode == "test" else data.val_mask
    feat = data.feat
    label = data.label
    dataloader = DataLoader(
            data.graph,
            torch.arange(data.graph.num_nodes()).to(device),
            sampler,
            device=device,
            batch_size=config.batch_size,
            use_uva=config.sample_mode == "uva",
            use_ddp=True,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

    with torch.no_grad() as no_grad:
        for layer_idx, layer in enumerate(_model.layers):
            num_col = (
                config.hid_size
                if layer_idx != len(_model.layers) - 1
                else _model.out_feats
            )
            
            y = torch.zeros(data.graph.num_nodes(), num_col, dtype=torch.float32, device=device)

            for input_nodes, output_nodes, blocks in dataloader:
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if layer_idx != len(_model.layers) - 1:
                    h = _model.dropout(h)
                # by design, our output nodes are contiguous
                if config.model == "gat":
                    h = h.flatten(1)

                # print(f"{input_nodes.shape=} {output_nodes.shape=} {y.shape=} {h.shape=} {config.model=}")
                y[output_nodes] = h
            feat = y
            dist.all_reduce(feat)

    pred = feat
    ypred = pred[mask]
    ylabel = label[mask]

    return MF.accuracy(
        ypred,
        ylabel,
        task="multiclass",
        num_classes=data.num_classes,
    ).item()

def inference_minibatch(config: Config, data: Dataset, model: DDP, mode):
    assert(mode in ["test", "eval"])
    device = torch.cuda.current_device()
    sampler = NeighborSampler(config.fanouts)
    ddp_module: MODEL_TYPES = model.module
    feat = data.feat
    label = data.label
    mask = data.test_mask if mode == "test" else data.val_mask
    dataloader = DataLoader(
            data.graph,
            mask,
            sampler,
            device=device,
            batch_size=config.batch_size,
            use_uva=config.sample_mode == "uva",
            use_ddp=True,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

    model.eval()
    ys = []
    y_hats = []
    for input_nodes, output_nodes, blocks in dataloader:
        x = feat[input_nodes]
        batch_pred = model(blocks, x)
        batch_label = label[output_nodes]
        ys.append(batch_label)
        y_hats.append(batch_pred)

    acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=data.num_classes)
    dist.all_reduce(acc, op=dist.ReduceOp.AVG)
    return acc.item()

def can_use_full(config: Config, data: Dataset):
    device = torch.cuda.current_device()
    gpu_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_reserved(device)
    feat_size = data.graph.num_nodes() * data.in_feats * 4
    return gpu_memory > feat_size * 2

def evaluate(config: Config, data: Dataset, model: DDP):
    # return inference_minibatch(config, data, model, "eval")
    if can_use_full(config, data):
        return inference_full(config, data, model, "eval")
    else:
        return inference_minibatch(config, data, model, "eval")

def test(config: Config, data: Dataset, model: DDP):
    if can_use_full(config, data):
        return inference_full(config, data, model, "test")
    else:
        return inference_minibatch(config, data, model, "test")