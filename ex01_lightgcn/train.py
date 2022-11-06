#! /usr/bin/env python

import argparse
import torch
import torch.nn as nn
import dgl

from dataloader import GowallaEdge, GowallaCheckIns
from model import SimpleGCN, LightGCN

NEPOCHS = 10

opts = argparse.ArgumentParser("Arguments for GCN model")
opts.add_argument("-b", "--batch-size", type=int, default=256, help="dataset batch size")
opts.add_argument("-l", "--learning-rate", type=float, default=1e-3, help="default learning rate")
opts.add_argument("-n", "--nepochs", type=int, default=100, help="default training epochs")
opts.add_argument("-s", "--embedding-size", type=int, default=32, help="embedding size")
opts.add_argument('-m', '--num-layers', nargs='+', help='number of gcn layers', required=True)


def train(args):
    dataset = GowallaCheckIns()
    graph: dgl.DGLGraph = dataset.graph
    graph = graph #.to('cuda:0')
    sampler = dgl.sampling.PinSAGESampler(graph, "user", "item", 3, 0.5, 200, 10)
    seeds = torch.LongTensor([0, 1, 2])
    froniter = sampler(seeds)
    # model = SimpleGCN(len(graph.nodes), 32)
    model = LightGCN(args.num_layers,
                     len(graph.nodes("user")), len(graph.nodes("item")),
                     list(map(int, args.num_layers)))
    ce = nn.CrossEntropyLoss(reduction="mean")
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for _ in range(args.nepochs):
        u, v = froniter.all_edges(form='uv')
        batch = dgl.heterograph({
            ('user', 'u2i', 'item'): (u, v),
            ('item', 'i2u', 'user'): (v, u),
        })
        res = model(batch)
        user = batch.nodes("user")
        item = batch.nodes("item")

        # add loss function
        loss1 = ce(res["user"], user)
        loss2 = ce(res["item"], item)
        loss = loss1 + loss2
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"### current user loss {loss1.item():12.6f}, item loss {loss2.item():12.6f} ###")


if __name__ == "__main__":
    args = opts.parse_args()
    train(args)