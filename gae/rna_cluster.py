# Train an graph-based model for RNA clustering

from typing import Optional
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import torch_geometric
import torch
import subprocess
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch.nn import functional as F



# def load_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
#     """
#     Load data from a file.
#     :param path: path to the file
#     :param nrows: number of rows to read
#     :return: pandas DataFrame
#     """
#     df = pd.read_csv(path, sep='\t', header=None, names=['sequence', 'energy', 'graph'], nrows=nrows)
#     df['graph'] = df['graph'].apply(nx.from_edgelist)
#     return df

if __name__ == "__main__":
    test = One_RNA_Dataset(root='data/test')
