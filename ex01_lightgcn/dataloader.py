#! /usr/bin/env python

""" Graph dataloaders for Gowallan dataset
* Web link for the dataset https://snap.stanford.edu/data/loc-gowalla.html
"""

import os
from dgl.data import DGLDataset
from torch.utils.data import Dataset
import pandas as pd
import urllib.request


class Gowalla(DGLDataset, Dataset):
    def __init__(self):
        # super().__init__()

        self.edge_url = "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
        self.checkin_url = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"

        if not self.has_cache():
            self.download()
            self.process()
            self.save()
        else:
            self.load()

    def has_cache(self):
        return os.path.exists(
            os.path.join(".datasrc", "loc-gowalla_edges.pkl")
        ) and os.path.exists(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.pkl")
        )

    def download(self):
        if not os.path.exists(".datasrc"):
            os.mkdir(".datasrc")
        print("# Download dataset...")
        urllib.request.urlretrieve(self.edge_url, os.path.join(".datasrc", "loc-gowalla_edges.txt.gz"))
        urllib.request.urlretrieve(self.checkin_url, os.path.join(".datasrc", "loc-gowalla_totalCheckins.txt.gz"))
        print("# Download finished")

    def process(self):
        print("# Preprocess dataset...")
        self.edge_data = pd.read_csv(
            os.path.join(".datasrc", "loc-gowalla_edges.txt.gz"),
            delim_whitespace=True,
            compression="gzip"
        )
        self.checkin_data = pd.read_csv(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.txt.gz"),
            delim_whitespace=True,
            compression="gzip"
        )
        print("# Data preprocess finished")

    def save(self):
        # serialize the dataframe
        self.edge_data.to_pickle(
            os.path.join(".datasrc", "loc-gowalla_edges.pkl")
        )
        self.checkin_data.to_pickle(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.pkl")
        )

    def load(self):
        self.edge_data = pd.read_pickle(
            os.path.join(".datasrc", "loc-gowalla_edges.pkl")
        )
        self.checkin_data = pd.read_pickle(
            os.path.join(".datasrc", "loc-gowalla_totalCheckins.pkl")
        )

    def __len__(self):
        return len(self.edge_data)

    def __getitem__(self, idx):
        return None

if __name__ == "__main__":
    dataset = Gowalla()
