import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.conv import GraphConv
from util import Config

activation = F.relu

class SAGE(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        num_layers: int,
        out_feats: int,
        dropout: float = 0.3
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
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

class GCN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        num_layers: int,
        out_feats: int,
        dropout: float = 0.3
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
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

class GAT(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        num_layers: int,
        out_feats: int,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
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