#! /usr/bin/env python

""" Graph dataloaders for Gowallan dataset
* Web link for the dataset https://snap.stanford.edu/data/loc-gowalla.html
"""

import os
import pickle

import dgl
import numpy as np
import tqdm
from dgl.data import DGLDataset
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_categorical
import urllib.request



def _series_to_tensor(series):
    if is_categorical(series):
        return torch.LongTensor(series.cat.codes.values.astype('int64'))
    else:       # numeric
        return torch.FloatTensor(series.values)

# Create graph from pandas, referring to the link
# https://github.com/dmlc/dgl/blob/17f1432ab2c74bed54df863be48e23b4113cbb37/examples/pytorch/pinsage/builder.py#L3
class PandasGraphBuilder(object):
    """ Build DGL graph based on pandas data

    """
    def __init__(self):
        pass

    @staticmethod
    def build_graph_from_edge(pdtable):
        u = pdtable[0].to_numpy()
        v = pdtable[1].to_numpy()
        # print(u, v)
        graph = dgl.graph(data=(u, v))
        # No need to iterate the edges since it's large graph
        # for _, row in tqdm.tqdm(pdtable.groupby(np.arange(len(pdtable))//bs)):
        #     start, end = row[0], row[1]
        #     graph.add_edges(start.to_numpy(), end.to_numpy())
        #     # uncomment just for test
        #     break
        return graph

    @staticmethod
    def build_graph_from_checkin(pdtable):
        u = pdtable[0].to_numpy()
        v = pdtable[4].to_numpy()
        # should be heterograph, need to be modified
        graph = dgl.heterograph({
            ('user', 'checkin', 'item'): (u, v)
        })
        return graph


class GowallaEdge(DGLDataset, Dataset):
    """ Snap dataset describing users sharing their locations, see
    https://snap.stanford.edu/data/loc-gowalla.html
    """
    def __init__(self):
        # super().__init__()

        self.edge_url = "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"

        if not self.has_cache():
            self.download()
            self.process()
            self.save()
        else:
            self.load()
        print(self.edge_data)

    def has_cache(self):
        return os.path.exists(
            os.path.join(".datasrc", "loc-gowalla_edges.pkl")
        )

    def download(self):
        if not os.path.exists(".datasrc"):
            os.mkdir(".datasrc")
        print("# Download dataset...")
        urllib.request.urlretrieve(self.edge_url, os.path.join(".datasrc", "loc-gowalla_edges.txt.gz"))
        print("# Download finished")

    def process(self):
        print("# Preprocess dataset...")
        self.edge_data = pd.read_csv(
            os.path.join(".datasrc", "loc-gowalla_edges.txt.gz"),
            header=None,
            delim_whitespace=True,
            compression="gzip"
        )
        builder = PandasGraphBuilder()
        self.graph = builder.build_graph_from_edge(self.edge_data)
        print("# Data preprocess finished")

    def save(self):
        # serialize the dataframe
        self.edge_data.to_pickle(
            os.path.join(".datasrc", "loc-gowalla_edges.pkl")
        )
        with open(os.path.join(".datasrc", "loc-gowalla.pkl"), "wb") as fout:
            pickle.dump(self.graph, fout)

    def load(self):
        self.edge_data = pd.read_pickle(
            os.path.join(".datasrc", "loc-gowalla_edges.pkl")
        )
        with open(os.path.join(".datasrc", "loc-gowalla.pkl"), "rb") as fin:
            self.graph = pickle.load(fin)

    def __len__(self):
        return len(self.edge_data)

    def __getitem__(self, idx):
        return None

class GowallaCheckIns(DGLDataset, Dataset):
    def __init__(self):
        # super().__init__()

        self.checkin_url = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
        if not self.has_cache():
            self.download()
            self.process()
            self.save()
        else:
            self.load()

        print("# Data preprocess finished")

    def has_cache(self):
        return os.path.exists(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.pkl")
        )

    def download(self):
        if not os.path.exists(".datasrc"):
            os.mkdir(".datasrc")
        print("# Download dataset...")
        urllib.request.urlretrieve(self.checkin_url, os.path.join(".datasrc", "loc-gowalla_totalCheckins.txt.gz"))
        print("# Download finished")

    def process(self):
        print("# Preprocess dataset...")
        self.checkin_data = pd.read_csv(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.txt.gz"),
            header=None,
            delim_whitespace=True,
            compression="gzip"
        )
        builder = PandasGraphBuilder()
        self.graph = builder.build_graph_from_checkin(self.checkin_data)
        print("# Data preprocess finished")

    def save(self):
        # serialize the dataframe
        self.checkin_data.to_pickle(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.pkl")
        )
        with open(os.path.join(".datasrc", "loc-gowalla_edges.pkl"), "wb") as fout:
            pickle.dump(self.graph, fout)

    def load(self):
        self.checkin_data = pd.read_pickle(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.pkl")
        )
        with open(os.path.join(".datasrc", "loc-gowalla_edges.pkl"), "rb") as fin:
            self.graph = pickle.load(fin)

    def __len__(self):
        return len(self.edge_data)

    def __getitem__(self, idx):
        return None

if __name__ == "__main__":
    # dataset = GowallaEdge()
    # builder = PandasGraphBuilder()
    # graph: dgl.DGLGraph = builder.build_graph_from_edge(dataset.edge_data)
    # print(graph)
    # graph.set_batch_num_nodes(128)
    # print(dgl.batch([graph]))
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    # dataloader = dgl.dataloading.DataLoader(
    #     graph, graph.nodes, sampler,
    #     batch_size=1024,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=4)
    # input_nodes, output_nodes, blocks = next(iter(dataloader))
    # print(blocks)
    dataset = GowallaCheckIns()
    graph: dgl.DGLGraph = dataset.graph
    print(graph)