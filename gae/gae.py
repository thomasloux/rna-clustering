# Cluster RNA using secondary structure
# Learning graph embeddings with Graph Autoencoders (torch geometric)

import torch
import numpy as np
import networkx as nx
import pandas as pd
import os.path as osp
from typing import Sequence, Optional, Tuple

import torch_geometric
from torch_geometric.nn.models.autoencoder import GAE
from torch_geometric import nn
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
import subprocess
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from time import time


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, number_channel: int) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = nn.GCNConv(hidden_channels, number_channel)

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


def get_vanilla_model(hidden_channels) -> GAE:
    """
    Get vanilla GAE model
    """
    model = GAE(
        Encoder(in_channels=4, hidden_channels=hidden_channels, number_channel=hidden_channels),
    )
    return model

def base_pair_distance(structure1: Data, structure2: Data) -> float:
    """
    Compute the base pair distance between two RNA structures
    it corresponds to the symetric difference between the two sets of base pairs

    :param structure1: first RNA structure
    :param structure2: second RNA structure
    :return: base pair distance
    """
    edges1 = set(structure1.edge_index.t())
    edges2 = set(structure2.edge_index.t())
    return float(len(edges1.symmetric_difference(edges2)))

def get_couples(data: One_RNA_Dataset, n: int) -> list[tuple[Data, Data]]:
    """
    Generate n couples of graphs from a dataset
    """
    rng = np.random.default_rng()
    random_index = rng.integers(0, len(data), size=(n, 2))
    return [(data[i], data[j]) for i, j in random_index]

def graph_embedding_from_node_embedding(node_embedding: torch.Tensor) -> torch.Tensor:
    """
    Compute the graph embedding from the node embedding
    """
    return torch.mean(node_embedding, dim=0)

def get_trained_model(epoch: int, model: GAE) -> GAE:
    """
    Return a trained model

    :param epoch: number of epochs for training
    :param model: model to train
    :return: trained model
    """

    # Load data
    data = One_RNA_Dataset(root='data/test')
    data_batch = DataLoader(data, batch_size=128, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    def train(graph):
        model.train()
        optimizer.zero_grad()
        # Compute reconstuction loss for graph
        z = model.encode(graph.x, graph.edge_index)
        loss = model.recon_loss(z, graph.edge_index)
        loss.backward()
        optimizer.step()
        return loss

    for e in tqdm(range(epoch)):
        for batch in data_batch:
            train(batch)
        # if e % (epoch//10) == 0:
        #      print(f"Epoch: {e:03d}, Loss: {loss:.4f}")
    return model

def get_couple_trained_model(epoch: int, model: GAE, distance, alpha: float = 1) -> Tuple[GAE, list[float]]:
    """
    Return a trained model

    :param epoch: number of epochs for training
    :param model: model to train
    :param distance: distance function to use in the loss
    :param alpha: weight of the distance loss
    :return: trained model
    """

    # Load data
    data = One_RNA_Dataset(root='data/test')
    data_batch = get_couples(data, n=100)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    def train(data):
        graph1, graph2 = data
        model.train()
        optimizer.zero_grad()

        # Compute reconstuction loss for graph1
        z1 = model.encode(graph1.x, graph1.edge_index)
        loss1 = model.recon_loss(z1, graph1.edge_index)

        # Compute reconstuction loss for graph2
        z2 = model.encode(graph2.x, graph2.edge_index)
        loss2 = model.recon_loss(z2, graph2.edge_index)

        # Compute loss distance and scalar product between embeddings
        graph1_embedding = graph_embedding_from_node_embedding(z1)
        graph2_embedding = graph_embedding_from_node_embedding(z2)
        scalar_distance = torch.dot(graph1_embedding, graph2_embedding)
        distance_loss = (scalar_distance - torch.tensor(distance(graph1, graph2)))**2

        # Total loss
        loss = loss1 + loss2 + alpha * distance_loss
        loss.backward()
        optimizer.step()
        return loss

    loss_record = []
    for e in tqdm(range(epoch)):
        loss = 0
        for batch in data_batch:
            loss += train(batch)
        loss_record.append(float(loss.detach()))
        # if e % (epoch//10) == 0:
        #      print(f"Epoch: {e:03d}, Loss: {loss:.4f}")
    return model, loss_record


def test_model(model, dataset):
    """
    Compute total loss on a dataset
    It only takes into account the positive examples (edges that are present in the graph)
    """
    model.eval()
    data_loader = DataLoader(dataset, batch_size=128)
    loss = 0
    with torch.no_grad():
        for batch in data_loader:
            z = model.encode(batch.x, batch.edge_index)
            # provide negative samples from the graph
            negative_edge_index = torch_geometric.utils.negative_sampling(
                edge_index=batch.edge_index, num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_index.size(1))
            loss += model.recon_loss(z, batch.edge_index, negative_edge_index)
    return loss/len(dataset)

def plot_loss(loss_record: Sequence[float]) -> None:
    """
    Plot the loss during training
    """
    plt.plot(loss_record[1:])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss during training')
    plt.savefig('loss.png')

if __name__ == "__main__":
    model = get_vanilla_model(hidden_channels=32)
    model, loss = get_couple_trained_model(epoch=1, model=model, distance=base_pair_distance)
    print(test_model(model, One_RNA_Dataset(root='data/test')))

