#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, NodeEmbedding

class SimpleGCN(nn.Module):
    """ LightGCN module

    """
    def __init__(self, num_nodes, hiddens):
        super().__init__()
        self.embed = NodeEmbedding(num_nodes, hiddens[0])
        self.conv1 = GraphConv(32, 32, norm='both', weight=True, bias=True)
        self.conv2 = GraphConv(32, 32, norm='both', weight=True, bias=True)
        self.conv3 = GraphConv(32, 1, norm='both', weight=True, bias=True)

    def forward(self, g:dgl.DGLGraph):
        x = self.embed(g.nodes)
        x = F.relu(self.conv1(g, x))
        x = F.relu(self.conv2(g, x))
        x = self.conv3(x).squeeze(-1)
        return x