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
opts.add_argument("-b", "--batch-size", type=int, default=256, help="dataset batch size")
opts.add_argument("-l", "--learning-rate", type=float, default=1e-3, help="default learning rate")
opts.add_argument("-n", "--nepochs", type=int, default=100, help="default training epochs")
opts.add_argument('-m','--num-layers', nargs='+', help='number of gcn layers', required=True)


def train(args):
    dataset = GowallaEdge()
    graph: dgl.DGLGraph = dataset.graph
    graph = graph.to('cuda:0')
    model = SimpleGCN(len(graph.nodes), 32)
    ce = nn.CrossEntropyLoss(reduction="mean")
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for _ in range(args.nepochs):
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
    args = opts.parse_args()
    train(args)