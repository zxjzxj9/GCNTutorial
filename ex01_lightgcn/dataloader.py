#! /usr/bin/env python

""" Graph dataloaders for Gowallan dataset
* Web link for the dataset https://snap.stanford.edu/data/loc-gowalla.html
"""

import os

import tqdm
from dgl.data import DGLDataset
import torch
from torch.utils.data import Dataset
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
    def __init__(self, pdtable: pd.DataFrame):
        self.pdtable = pdtable

    def build_graph(self):

        for row in tqdm.tqdm(self.pdtable.iterrows()):
            pass

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
        print("# Data preprocess finished")

    def save(self):
        # serialize the dataframe
        self.edge_data.to_pickle(
            os.path.join(".datasrc", "loc-gowalla_edges.pkl")
        )

    def load(self):
        self.edge_data = pd.read_pickle(
            os.path.join(".datasrc", "loc-gowalla_edges.pkl")
        )

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
        print("# Data preprocess finished")

    def save(self):
        # serialize the dataframe
        self.checkin_data.to_pickle(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.pkl")
        )

    def load(self):
        self.checkin_data = pd.read_pickle(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.pkl")
        )

    def __len__(self):
        return len(self.edge_data)

    def __getitem__(self, idx):
        return None

if __name__ == "__main__":
    dataset = GowallaEdge()

