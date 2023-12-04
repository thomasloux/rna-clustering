# Cluster RNA using secondary structure
# Learning graph embeddings with Graph Autoencoders (torch geometric)

import torch
import numpy as np
import pandas as pd
import os
import os.path as osp
from typing import Sequence, Optional, Tuple, Union

import torch_geometric
from torch_geometric.nn.models.autoencoder import GAE
from torch_geometric import nn
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from torch import Tensor
import subprocess
from tqdm import tqdm

import matplotlib.pyplot as plt

from time import time

from data import One_RNA_Dataset, PairDataset


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = nn.GCNConv(hidden_channels, hidden_channels)
        self.conv4 = nn.GCNConv(hidden_channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index)
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
        Encoder(in_channels=4, hidden_channels=hidden_channels),
    )
    return model


def get_trained_model(epoch: int, model: GAE, root: str = "data/test") -> GAE:
    """
    Return a trained model

    :param epoch: number of epochs for training
    :param model: model to train
    :return: trained model
    """

    # Load data
    data = One_RNA_Dataset(root=root)
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

def get_couple_trained_model(epoch: int, model: GAE, distance, alpha: float = 1, root: str = "data/test") -> Tuple[GAE, list[float]]:
    """
    Return a trained model

    :param epoch: number of epochs for training
    :param model: model to train
    :param distance: distance function to use in the loss
    :param alpha: weight of the distance loss
    :return: trained model
    """

    # Load data
    data = One_RNA_Dataset(root=root)
    data = PairDataset(data, data, sample=True)
    data_batch = DataLoader(data, batch_size=16, shuffle=False, follow_batch=["x_1", "x_2"])
    print(len(data))
    #### ADD TRANSFORM HERE ###
    # Normalize for the distance, maybe 2*size of the sequence
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    def train(data_batch):
        """
        Train the model on a batch of data

        data_batch contains :
        - x_1: node features for graph 1, format [sum(num_nodes) in the batch, num_features for graph_1]
        - edge_index_1: edge index for graph 1, format [2, sum(num_edges) in the batch for graph_1]
        - x_2: node features for graph 2
        - edge_index_2: edge index for graph 2
        - x_1_batch: batch index for graph 1, format [sum(num_nodes) in the batch]
        - x_2_batch: batch index for graph 2
        """
        model.train()
        optimizer.zero_grad()

        # Compute reconstuction loss for graph1
        z1 = model.encode(data_batch.x_1, data_batch.edge_index_1)
        # Renormalize the embeddings
        z1_renorm = z1 / torch.norm(z1, dim=1).unsqueeze(1)
        loss1 = model.recon_loss(z1_renorm, data_batch.edge_index_1)

        # Compute reconstuction loss for graph2
        z2 = model.encode(data_batch.x_2, data_batch.edge_index_2)
        # Renormalize the embeddings
        z2_renorm = z2 / torch.norm(z2, dim=1).unsqueeze(1)
        loss2 = model.recon_loss(z2_renorm, data_batch.edge_index_2)

        # Compute loss distance and scalar product between embeddings
        graph1_embedding = torch_geometric.nn.pool.global_mean_pool(z1, data_batch.x_1_batch)
        graph2_embedding = torch_geometric.nn.pool.global_mean_pool(z2, data_batch.x_2_batch)
        scalar_product = - (graph1_embedding * graph2_embedding).sum(axis=1)  # Scalar product between embeddings
        # euclidean_distance = torch.norm(graph1_embedding - graph2_embedding, dim=1)  # Euclidean distance between embeddings
        #print("Euclidean distance", euclidean_distance.mean())
        #print("Scalar product", scalar_product.mean())
        distances = distance(
            data_batch.edge_index_1,
            data_batch.edge_index_2,
            batch_1=data_batch.x_1_batch,
            batch_2=data_batch.x_2_batch
        )
        #print("Distance", distances.mean())
        distance_loss = (scalar_product - distances)**2

        # Total loss
        loss = (loss1 + loss2) + alpha * distance_loss.mean()
        #print("Loss recontuction ", loss1 + loss2, "Distance loss", (alpha * distance_loss.mean()))
        loss.backward()
        optimizer.step()

        nb_graphs = data_batch.x_1_batch[-1] + 1
        return loss * nb_graphs, (loss1 + loss2) * nb_graphs, alpha * distance_loss.sum()

    total_loss_record = []
    total_loss_reconstruction_record = []
    total_loss_distance_record = []
    for e in tqdm(range(epoch)):
        loss = 0
        loss_reconstruction = 0
        loss_distance = 0
        for batch in data_batch:
            loss_batch, loss_reconstruction_batch, loss_distance_batch = train(batch)
            loss += loss_batch
            loss_reconstruction += loss_reconstruction_batch
            loss_distance += loss_distance_batch
        total_loss_record.append(float(loss.detach())/len(data))
        total_loss_reconstruction_record.append(float(loss_reconstruction.detach()/len(data)))
        total_loss_distance_record.append(float(loss_distance.detach())/len(data))
    
    print("Total loss", total_loss_record[-1])
    print("Reconstruction loss", total_loss_reconstruction_record[-1])
    print("Distance loss", total_loss_distance_record[-1])
        # if e % (epoch//10) == 0:
        #      print(f"Epoch: {e:03d}, Loss: {loss:.4f}")
    return model, [total_loss_record, total_loss_reconstruction_record, total_loss_distance_record]


def test_model_reconstruction(model, dataset):
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

def test_model_distance(model, dataset, distance):
    """
    Compute total loss on a dataset
    for the distance and scalar product between embeddings
    """
    model.eval()
    data_loader = DataLoader(dataset, batch_size=128, follow_batch=["x_1", "x_2"])
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            # Encode graphs
            z1 = model.encode(batch.x_1, batch.edge_index_1)
            z2 = model.encode(batch.x_2, batch.edge_index_2)

            # Graph embeddings
            graph1_embedding = torch_geometric.nn.pool.global_mean_pool(z1, batch.x_1_batch)
            graph2_embedding = torch_geometric.nn.pool.global_mean_pool(z2, batch.x_2_batch)

            # Scalar product between embeddings (predicted distance)
            scalar_distance = (graph1_embedding * graph2_embedding).sum(axis=1)

            # Real distance
            distances = distance(
                batch.edge_index_1,
                batch.edge_index_2,
                batch_1=batch.x_1_batch,
                batch_2=batch.x_2_batch
            )

            # Square Mean Error between predicted distance and true distance
            loss_batch = (scalar_distance - distances)**2

            # Losses history
            losses += loss_batch.detach().numpy().tolist()
    return losses


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

# if __name__ == "__main__":
#     model = get_vanilla_model(hidden_channels=32)
#     model, loss = get_couple_trained_model(epoch=1, model=model, distance=base_pair_distance)
#     print(test_model(model, One_RNA_Dataset(root='data/test')))

