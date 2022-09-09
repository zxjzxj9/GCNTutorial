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

    def forward(self, g:dgl.DGLGraph):
        x = self.embed(g.nodes())
        x = F.relu(self.conv1(g, x))
        x = F.relu(self.conv2(g, x))
        x = self.conv3(x).squeeze(-1)
        return x

# see this paper: https://arxiv.org/pdf/2002.02126.pdf
class NGCF(nn.Module):
    """ NGCF model, for comparison
    """
    def __init__(self, embed_dim, num_user, num_item):
        super().__init__()

        self.user_embed = NodeEmbedding(num_user, embed_dim, "user", init_func=init_func)
        self.item_embed = NodeEmbedding(num_item, embed_dim, "item", init_func=init_func)
        self.conv1 = HeteroGraphConv({
            "user": GraphConv(32, 32, norm='both', weight=True, bias=True),
            "checkin": SAGEConv(32, 32, aggregator_type='sum'),
            "item": GraphConv(32, 32, norm='both', weight=True, bias=True),
        })

    def forward(self, g: dgl.DGLGraph):
        user_vec = self.user_embed(g.nodes["user"])
        item_vec = self.item_embed(g.nodes["item"])

        return user_vec

class LightGCN(nn.Module):
    """ LightGVN model
    """
    def __init__(self):
        pass

    def forward(self, g:dgl.DGLGraph):
        pass


if __name__ == "__main__":
    pass