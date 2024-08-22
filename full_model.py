import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import GraphConv
from util import Dataset, Config
from typing import Union

activation = F.relu


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

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.layers.append(
                    GraphConv(in_feats, hid_feats, activation=activation)
                )
            elif layer_idx < num_layers - 1:
                self.layers.append(
                    GraphConv(hid_feats, hid_feats, activation=activation)
                )
            else:
                self.layers.append(GraphConv(hid_feats, out_feats))

    def forward(self, g, features) -> torch.Tensor:
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


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

    def forward(self, g, features) -> torch.Tensor:
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            h = h.flatten(1)
        return h


def evaluate(data: Dataset, model: GCN | GAT) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(data.graph, data.feat)
        logits = logits[data.val_mask]
        labels = data.label[data.val_mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def test(data: Dataset, model: Union[GCN, GAT]) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(data.graph, data.feat)
        logits = logits[data.test_mask]
        labels = data.label[data.test_mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
