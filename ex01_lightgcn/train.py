#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dataloader import GowallaEdge
from model import SimpleGCN

NEPOCHS = 10

def train(args):
    dataset = GowallaEdge()
    graph: dgl.DGLGraph = dataset.graph
    graph = graph.to('cuda:0')
    model = SimpleGCN(len(graph.nodes), 32)
    ce = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(NEPOCHS):
        logits = model(graph)
        target = graph.nodes()
        # add loss function
        loss = ce(logits, target)
        optim.zero_grad()
        loss.backward()
        optim.step()



if __name__ == "__main__":
    train()