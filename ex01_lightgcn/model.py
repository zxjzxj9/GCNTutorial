#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class LightGCN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass