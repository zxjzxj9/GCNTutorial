#! /usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dataloader import GowallaEdge
from model import SimpleGCN

NEPOCHS = 10

opts = argparse.ArgumentParser("Arguments for GCN model")
args = opts.parse_args()

def train(args):
    dataset = GowallaEdge()
    graph: dgl.DGLGraph = dataset.graph
    graph = graph.to('cuda:0')
    model = SimpleGCN(len(graph.nodes), 32)
    ce = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(NEPOCHS):
        res = model(graph)
        user = graph.nodes("user")
        item = graph.nodes("item")
        # add loss function
        loss1 = ce(res["user"], user)
        loss2 = ce(res["item"], item)
        loss = loss1 + loss2
        optim.zero_grad()
        loss.backward()
        optim.step()



if __name__ == "__main__":
    train()