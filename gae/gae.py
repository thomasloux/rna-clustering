# Cluster RNA using secondary structure
# Learning graph embeddings with Graph Autoencoders (torch geometric)

import torch
import numpy as np
import networkx as nx
import pandas as pd
from typing import Sequence, Optional

import torch_geometric
from torch_geometric.nn.models.autoencoder import GAE
from torch_geometric import nn
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch.nn import functional as F
import subprocess

from time import time

class One_RNA_InMemoryDataset(InMemoryDataset):
    """
    Dataset for one RNA sequence.

    Take one RNA sequence and generate a certain number of suboptimal structures. Transform each structure into a graph.
    Raw data : RNA sequence (only the sequence in the first line of the file)
    :param
    """
    nucleotides_correspondance = {
        'A': 0,
        'C': 1,
        'G': 2,
        'U': 3,
        'T': 3
    }


    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        """
        :param root: root directory where the dataset should be saved.
            This folder is split into raw_dir (downloaded dataset) and processed_dir (processed dataset).
        :param n: number of suboptimal structures to generate
        """
        super(One_RNA_Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        print(f"Root : {root}")
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['test.seq']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise NotImplementedError('Raw data not found.')

    def structure_to_data(self, structure: str, sequence: str, weight: Optional[float] = 1.0) -> Data:
        """
        Convert a RNA structure to a graph of type torch_geometric.data.Data.
        """
        ## Node features (shape : num_nodes x num_node_features)
        # One-hot encoding of the nucleotide
        x = torch.Tensor([self.nucleotides_correspondance[sequence[i]] for i in range(len(sequence))]).long()
        x = F.one_hot(x, 4).float()

        ## Edge index (shape : 2 x num_edges)
        # Edge index is a tensor of size 2 x num_edges where each column represents an edge.
        edge_index_strong = torch.tensor([range(len(sequence) - 1), range(1, len(sequence))], dtype=torch.long)
        edge_pairs = []
        stack = []

        for i, c in enumerate(structure):
            if c == '(':
                stack.append(i)
            elif c == ')':
                j = stack.pop()
                edge_pairs.extend([(i, j)])

        # Convert the list of edge pairs to a tensor
        edge_index_weak = torch.tensor(edge_pairs, dtype=torch.long).t()
        edge_pairs = torch.cat(
            [
                edge_index_strong,
                edge_index_strong.flip(0),
                edge_index_weak,
                edge_index_weak.flip(0)
            ],
            dim=1
        )
         
        data = Data(x=x, edge_index=edge_pairs)

        return data



    def generate_suboptimal_structures(self, input_file: str, n: int) -> list[str]:
        """
        Using ViennaRNA package to generate suboptimal structures.
        Note : we use CLI command because it offers more options than the python wrapper
        :param sequence: RNA sequence
        :param n: number of suboptimal structures to generate
        :return: list of suboptimal structures
        """
        with open(input_file, 'r') as f:
            sequence = f.readline().strip()
            assert set(sequence) <= set('ACGUT'), "Invalid sequence" 
        # --stochBT : stochastic backtracking
        # Return only the dot-bracket structure
        # --stochBT_en 
        # Return dot-bracket structure, energy and probability
        command = subprocess.run(["RNAsubopt", "--stochBT=" + str(n), "-i", input_file], capture_output=True)
        output = command.stdout.decode('utf-8').split('\n')
        # Remove first line (sequence) and last line (empty)
        output = output[1:-1]
        return output

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_paths[0], 'r') as f:
            sequence = f.readline().strip()

        structures = self.generate_suboptimal_structures(self.raw_paths[0], n=10)
        data_list = [self.structure_to_data(structure, sequence) for structure in structures]
        print(f"Number of structures : {len(data_list)}")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.data)

class One_RNA_Dataset(Dataset):
    """
    Dataset for one RNA sequence.

    Take one RNA sequence and generate a certain number of suboptimal structures. Transform each structure into a graph.
    Raw data : RNA sequence (only the sequence in the first line of the file)
    :param
    """
    nucleotides_correspondance = {
        'A': 0,
        'C': 1,
        'G': 2,
        'U': 3,
        'T': 3
    }


    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        """
        :param root: root directory where the dataset should be saved.
            This folder is split into raw_dir (downloaded dataset) and processed_dir (processed dataset).
        :param n: number of suboptimal structures to generate
        """
        super(One_RNA_Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        print(f"Root : {root}")
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['test.seq']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise NotImplementedError('Raw data not found.')

    def structure_to_data(self, structure: str, sequence: str, weight: Optional[float] = 1.0) -> Data:
        """
        Convert a RNA structure to a graph of type torch_geometric.data.Data.
        """
        ## Node features (shape : num_nodes x num_node_features)
        # One-hot encoding of the nucleotide
        x = torch.Tensor([self.nucleotides_correspondance[sequence[i]] for i in range(len(sequence))]).long()
        x = F.one_hot(x, 4).float()

        ## Edge index (shape : 2 x num_edges)
        # Edge index is a tensor of size 2 x num_edges where each column represents an edge.
        edge_index_strong = torch.tensor([range(len(sequence) - 1), range(1, len(sequence))], dtype=torch.long)
        edge_pairs = []
        stack = []

        for i, c in enumerate(structure):
            if c == '(':
                stack.append(i)
            elif c == ')':
                j = stack.pop()
                edge_pairs.extend([(i, j)])

        # Convert the list of edge pairs to a tensor
        edge_index_weak = torch.tensor(edge_pairs, dtype=torch.long).t()
        edge_pairs = torch.cat(
            [
                edge_index_strong,
                edge_index_strong.flip(0),
                edge_index_weak,
                edge_index_weak.flip(0)
            ],
            dim=1
        )
         
        data = Data(x=x, edge_index=edge_pairs)

        return data



    def generate_suboptimal_structures(self, input_file: str, n: int) -> list[str]:
        """
        Using ViennaRNA package to generate suboptimal structures.
        Note : we use CLI command because it offers more options than the python wrapper
        :param sequence: RNA sequence
        :param n: number of suboptimal structures to generate
        :return: list of suboptimal structures
        """
        with open(input_file, 'r') as f:
            sequence = f.readline().strip()
            assert set(sequence) <= set('ACGUT'), "Invalid sequence" 
        # --stochBT : stochastic backtracking
        # Return only the dot-bracket structure
        # --stochBT_en 
        # Return dot-bracket structure, energy and probability
        command = subprocess.run(["RNAsubopt", "--stochBT=" + str(n), "-i", input_file], capture_output=True)
        output = command.stdout.decode('utf-8').split('\n')
        # Remove first line (sequence) and last line (empty)
        output = output[1:-1]
        return output

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_paths[0], 'r') as f:
            sequence = f.readline().strip()

        structures = self.generate_suboptimal_structures(self.raw_paths[0], n=10)
        data_list = [self.structure_to_data(structure, sequence) for structure in structures]
        print(f"Number of structures : {len(data_list)}")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.data)



class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super(Encoder, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

def get_device() -> torch.device:
    """
    Get device for training (GPU or other acceleration if available)

    :return: torch.device object

    - MPS or Metal Performance Shaders is a GPU acceleration library for ARM Macs
    - Cuda is a GPU acceleration library for Nvidia GPUs
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def get_trained_model(epoch):
    # Parameters
    hidden_dim = 32

    # Load data
    data = One_RNA_Dataset(root='data/test')
    data_batch = DataLoader(data, batch_size=128, shuffle=True)

    # Model
    input_dim = 4
    encoder = Encoder(input_dim, hidden_dim)
    model = GAE(encoder)


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    def train(data):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        return loss

    def test(data):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
        return float(model.recon_loss(z, data.edge_index))

    for e in range(epoch):
        deb = time()
        loss = 0
        for batch in data_batch:
            loss += train(batch)
        if e % (epoch//10) == 0:
            print(f"Epoch: {e:03d}, Loss: {loss:.4f}, Time: {time() - deb:.4f}s")
    #print("Test provide " + str(test(graph)))
    return model

if __name__ == "__main__":
    dataset = One_RNA_Dataset(root='data/test')
