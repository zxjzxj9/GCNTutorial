#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, HeteroGraphConv, SAGEConv, NodeEmbedding


def init_func(embed):
    nn.init.uniform_(embed, -1e-3, 1e-3)
    return embed


class SimpleGCN(nn.Module):
    """ SimpleGCN model, for comparison
    """

    def __init__(self, num_nodes, hiddens):
        super().__init__()
        self.embed = NodeEmbedding(num_nodes, hiddens[0], 'node_emb', init_func=init_func)
        self.conv1 = GraphConv(32, 32, norm='both', weight=True, bias=True)
        self.conv2 = GraphConv(32, 32, norm='both', weight=True, bias=True)
        self.conv3 = GraphConv(32, 1, norm='both', weight=True, bias=True)

    def forward(self, g: dgl.DGLGraph):
        x = self.embed(g.nodes())
        x = F.relu(self.conv1(g, x))
        x = F.relu(self.conv2(g, x))
        x = self.conv3(x).squeeze(-1)
        return x


# see this paper: https://arxiv.org/pdf/2002.02126.pdf
class NGCF(nn.Module):
    """ NGCF model, for comparison
    """

    def __init__(self, embed_dim, num_user, num_item, nhiddens):
        super().__init__()

        self.user_embed = NodeEmbedding(num_user, embed_dim, "user", init_func=init_func)
        self.item_embed = NodeEmbedding(num_item, embed_dim, "item", init_func=init_func)
        for cin, cout in zip(nhiddens, nhiddens[1:]):
            self.gconv.append(
                HeteroGraphConv({
                    "checkin": SAGEConv(cin, cout, norm='mean', weight=False, bias=False),
            }))


    def forward(self, g: dgl.DGLGraph):
        vecs = {
            "user": self.user_embed(g.nodes("user")),
            "item": self.item_embed(g.nodes("item")),
        }
        x = vecs

        for conv in self.gconv:
            x = conv(g, vecs)
            # skip last layer
            if conv is not self.gconv[-1]:
                x = {k: F.leaky_relu(v) for k, v in x.items()}
        return x


class LightGCN(nn.Module):
    """ LightGVN model
    """

    def __init__(self, embed_dim, num_user, num_item, nhiddens):
        super(LightGCN, self).__init__()
        self.user_embed = NodeEmbedding(num_user, embed_dim, "user", init_func=init_func)
        self.item_embed = NodeEmbedding(num_item, embed_dim, "item", init_func=init_func)

        nhiddens.insert(0, embed_dim)
        self.gconv = nn.ModuleList()
        for cin, cout in zip(nhiddens, nhiddens[1:]):
            self.gconv.append(
                HeteroGraphConv({
                    "u2i": GraphConv(cin, cout, norm='both', weight=False, bias=False),
                    "i2u": GraphConv(cin, cout, norm='both', weight=False, bias=False),
                }))


    def forward(self, g: dgl.DGLGraph):
        vecs = {
            "user": self.user_embed(g.nodes("user")),
            "item": self.item_embed(g.nodes("item")),
        }
        x = vecs

        for conv in self.gconv:
            x = conv(g, vecs)
            # skip last layer
            if conv is not self.gconv[-1]:
                x = {k: v.relu() for k, v in x.items()}
        return x


if __name__ == "__main__":
    u = torch.randint(low=0, high=16, size=(32,))
    v = torch.randint(low=0, high=16, size=(32,))
    graph = dgl.heterograph({
        ('user', 'u2i', 'item'): (u, v),
        ('item', 'i2u', 'user'): (v, u)
    })
    print(graph.nodes("user"))
    print(graph.nodes("item"))
    ce = nn.CrossEntropyLoss(reduction="mean")
    model = LightGCN(32, 16, 16, [32, 32, 32])
    ret = model(graph)
    user = graph.nodes("user")
    item = graph.nodes("item")
    user_vec = ret["user"]
    item_vec = ret["item"]
    # print(user_vec.softmax(-1))
    # print(item_vec.softmax(-1))
    print("loss1: {:12.6f}, loss2: {:12.6f}".format(ce(user_vec, user), ce(item_vec, item)))

    sampler = dgl.sampling.PinSAGESampler(graph, "user", "item", 3, 0.5, 200, 10)
    seeds = torch.LongTensor([0, 1, 2])
    froniter = sampler(seeds)
    for _ in range(10):
        batch = froniter.all_edges(form='uv')
        print(batch)