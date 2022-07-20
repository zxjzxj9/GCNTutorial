#! /usr/bin/env python

""" Graph dataloaders for Gowallan dataset
* Web link for the dataset https://snap.stanford.edu/data/loc-gowalla.html
"""

import os
from dgl.data import DGLDataset
import urllib.request


class Gowalla(DGLDataset):
    def __init__(self):
        super().__init__()

        self.edge_url = "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
        self.checkin_url = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"

    def has_cache(self):
        pass

    def download(self):
        if not os.path.exists(".datasrc"):
            os.mkdir(".datasrc")
        urllib.request.urlretrieve(self.edge_url, os.path.join(".datasrc", "loc-gowalla_edges.txt.gz"))
        urllib.request.urlretrieve(self.checkin_url, os.path.join(".datasrc", "loc-gowalla_totalCheckins.txt.gz"))

    def process(self):
        pass

    def save(self):
        pass

    def load(self):
        pass