#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

class LightGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(32, 32, norm='both', weight=True, bias=True)

    def forward(self, x):
        pass