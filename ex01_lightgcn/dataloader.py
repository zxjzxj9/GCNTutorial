#! /usr/bin/env python

""" Graph dataloaders for Gowallan dataset
* Web link for the dataset https://snap.stanford.edu/data/loc-gowalla.html
"""

from dgl.data import DGLDataset

class Gowalla(DGLDataset):
    def __init__(self):
        super().__init__()

        self.edge_url = "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
        self.checkin_url = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"

    def has_cache(self):
        pass

    def download(self):
        pass

    def process(self):
        pass

    def save(self):
        pass

    def load(self):
        pass