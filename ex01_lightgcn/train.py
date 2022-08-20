#! /usr/bin/env python
import dgl

from dataloader import GowallaEdge
from model import SimpleGCN

NEPOCHS = 10

def train(args):
    dataset = GowallaEdge()
    graph: dgl.DGLGraph = dataset.graph
    graph = graph.to('cuda:0')
    model = SimpleGCN(len(graph.nodes), 32)
    for _ in range(NEPOCHS):
        pass


if __name__ == "__main__":
    train()