import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.conv import GraphConv

from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)

import torchmetrics.functional as MF
from util import Dataset, Config, get_load_compute_cores
from typing import Union

activation = F.relu


class SAGE(nn.Module):
    def __init__(
        self,
        config: Config,
        in_feats: int,
        hid_feats: int,
        num_layers: int,
        out_feats: int,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)
        self.hid_feats = hid_feats
        self.out_feats = out_feats

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.layers.append(
                    SAGEConv(in_feats, hid_feats, "mean", activation=activation)
                )
            elif layer_idx < num_layers - 1:
                self.layers.append(
                    SAGEConv(hid_feats, hid_feats, "mean", activation=activation)
                )
            else:
                self.layers.append(SAGEConv(hid_feats, out_feats, "mean"))

    def forward(self, blocks, feat: torch.Tensor) -> torch.Tensor:
        h = feat
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)  # len(blocks) = 1
            if l != len(self.layers) - 1:
                h = self.dropout(h)
        return h

    def inference(
        self, batch_size: int, g: dgl.DGLGraph, feat: torch.Tensor
    ) -> torch.Tensor:
        loader_cores, compute_cores = get_load_compute_cores()

        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler,
            device="cpu",
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=len(loader_cores),
        )

        with torch.no_grad() as no_grad, dataloader.enable_cpu_affinity(
            loader_cores=loader_cores,
            compute_cores=compute_cores,
            verbose=False,
        ) as cpu_affin:
            for layer_idx, layer in enumerate(self.layers):
                num_col = (
                    self.hid_feats
                    if layer_idx != len(self.layers) - 1
                    else self.out_feats
                )
                y = torch.empty(g.num_nodes(), num_col, dtype=feat.dtype)

                for input_nodes, output_nodes, blocks in dataloader:
                    x = feat[input_nodes]
                    h = layer(blocks[0], x)  # len(blocks) = 1
                    if layer_idx != len(self.layers) - 1:
                        h = self.dropout(h)
                    # by design, our output nodes are contiguous
                    y[output_nodes[0] : output_nodes[-1] + 1] = h
                feat = y
            return y

class GCN(nn.Module):
    def __init__(
        self,
        config: Config,
        in_feats: int,
        hid_feats: int,
        num_layers: int,
        out_feats: int,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)
        self.hid_feats = hid_feats
        self.out_feats = out_feats

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.layers.append(
                    GraphConv(in_feats, hid_feats, "mean", activation=activation)
                )
            elif layer_idx < num_layers - 1:
                self.layers.append(
                    GraphConv(hid_feats, hid_feats, "mean", activation=activation)
                )
            else:
                self.layers.append(GraphConv(hid_feats, out_feats, "mean"))

    def forward(self, blocks, feat: torch.Tensor) -> torch.Tensor:
        h = feat
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)  # len(blocks) = 1
            if l != len(self.layers) - 1:
                h = self.dropout(h)
        return h

    def inference(
        self, batch_size: int, g: dgl.DGLGraph, feat: torch.Tensor
    ) -> torch.Tensor:
        loader_cores, compute_cores = get_load_compute_cores()

        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler,
            device="cpu",
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=len(loader_cores),
        )

        with torch.no_grad() as no_grad, dataloader.enable_cpu_affinity(
            loader_cores=loader_cores,
            compute_cores=compute_cores,
            verbose=False,
        ) as cpu_affin:
            for layer_idx, layer in enumerate(self.layers):
                num_col = (
                    self.hid_feats
                    if layer_idx != len(self.layers) - 1
                    else self.out_feats
                )
                y = torch.empty(g.num_nodes(), num_col, dtype=feat.dtype)

                for input_nodes, output_nodes, blocks in dataloader:
                    x = feat[input_nodes]
                    h = layer(blocks[0], x)  # len(blocks) = 1
                    if layer_idx != len(self.layers) - 1:
                        h = self.dropout(h)
                    # by design, our output nodes are contiguous
                    y[output_nodes[0] : output_nodes[-1] + 1] = h
                feat = y
            return y
        
class GAT(nn.Module):
    def __init__(
        self,
        config: Config,
        in_feats: int,
        hid_feats: int,
        num_layers: int,
        out_feats: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(config.dropout)
        hid_feats = int(hid_feats / num_heads)
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.layers.append(
                    GATConv(
                        in_feats=in_feats,
                        out_feats=hid_feats,
                        num_heads=num_heads,
                        activation=activation,
                        allow_zero_in_degree=False,
                    )
                )
            elif layer_idx < num_layers - 1:
                self.layers.append(
                    GATConv(
                        in_feats=hid_feats * num_heads,
                        out_feats=hid_feats,
                        num_heads=num_heads,
                        activation=activation,
                        allow_zero_in_degree=False,
                    )
                )
            else:
                self.layers.append(
                    GATConv(
                        in_feats=hid_feats * num_heads, out_feats=out_feats, num_heads=1
                    )
                )

    def forward(self, blocks, features: torch.Tensor) -> torch.Tensor:
        h = features
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.dropout(h)
            h = h.flatten(1)
        return h

    def inference(
        self, batch_size: int, g: dgl.DGLGraph, feat: torch.Tensor
    ) -> torch.Tensor:
        loader_cores, compute_cores = get_load_compute_cores()

        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler,
            device="cpu",
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=len(loader_cores),
        )

        with torch.no_grad() as no_grad, dataloader.enable_cpu_affinity(
            compute_cores=loader_cores,
            loader_cores=compute_cores,
            verbose=False,
        ) as cpu_affin:
            for layer_idx, layer in enumerate(self.layers):
                num_col = (
                    self.hid_feats * self.num_heads
                    if layer_idx != len(self.layers) - 1
                    else self.out_feats
                )
                y = torch.empty(g.num_nodes(), num_col, dtype=feat.dtype)

                for input_nodes, output_nodes, blocks in dataloader:
                    x = feat[input_nodes]
                    h = layer(blocks[0], x)  # len(blocks) = 1
                    if layer_idx != len(self.layers) - 1:
                        h = self.dropout(h)

                    h = h.flatten(1)
                    # by design, our output nodes are contiguous
                    y[output_nodes[0] : output_nodes[-1] + 1] = h
                feat = y
            return y


def evaluate(data: Dataset, model: SAGE | GAT):
    config = model.config
    model.eval()
    sampler = NeighborSampler(config.fanouts)
    dataloader = DataLoader(
        data.graph,
        torch.arange(data.graph.num_nodes()),
        sampler,
        device="cpu",
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=torch.get_num_threads(),
    )
    ylabel = []
    ypred = []
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = data.feat[input_nodes]
            yl = data.label[output_nodes]
            ylabel.append(yl)
            ypred.append(model(blocks, x))

    return MF.accuracy(
        torch.cat(ypred),
        torch.cat(ylabel),
        task="multiclass",
        num_classes=data.num_classes,
    ).item()


def test(data: Dataset, model: Union[SAGE, GAT]):
    model.eval()
    config = model.config
    pred = model.inference(config.batch_size, data.graph, data.feat)
    ypred = pred[data.test_mask]
    ylabel = data.label[data.test_mask]

    return MF.accuracy(
        ypred,
        ylabel,
        task="multiclass",
        num_classes=data.num_classes,
    ).item()
