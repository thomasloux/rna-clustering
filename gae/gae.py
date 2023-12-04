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
from torch.utils.data import random_split
from torch import Tensor
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from data import One_RNA_Dataset, PairDataset
import logging


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = nn.GCNConv(hidden_channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x



def get_vanilla_model(hidden_channels) -> GAE:
    """
    Get vanilla GAE model
    """
    model = GAE(
        Encoder(in_channels=4, hidden_channels=hidden_channels),
    )
    return model

def get_couple_trained_model(
    epoch: int,
    model: GAE,
    distance,
    save_folder: str,
    alpha: float = 0.1,
    root: str = "data/test",
    distance_from_embedding: str = "euclidean",
    distance_loss: str = "L2",
    distance_loss_only: bool = False,
    device: str = "cpu") -> Tuple[GAE, list[float]]:
    """
    Return a trained model

    Params:
    - epoch: number of epochs for training
    - model: model to train
    - distance: function to compute distance between graphs
    - save_folder: folder where to save the model
    - alpha: weight of the distance loss in the total loss
    - root: root directory where the dataset should be saved.
        This folder is split into raw_dir (downloaded dataset) and processed_dir (processed dataset).
    - distance_from_embedding: distance to use between embeddings (euclidean or scalar_product)
    - distance_loss: loss to use for the distance between embeddings and predicted distance (L2 or L1)
    - distance_loss_only: if True, only the distance loss is used for training
    - device: device for training (GPU or other acceleration if available)

    :return: trained model
    """

    # Check arguments
    if distance_from_embedding not in ["euclidean", "scalar_product"]:
        raise ValueError("distance_from_embedding must be either euclidean or scalar_product")
    if distance_loss not in ["L2", "L1"]:
        raise ValueError("distance_loss must be either L2 or L1")

    # Check if model already exists
    # if os.path.exists(os.path.join("models", save_folder)) and not os.path.isdir(os.path.join("models", save_folder)):
    #     raise ValueError("Model already exists")
    # os.mkdir(os.path.join("models", save_folder))

    device = torch.device(device)
    model = model.to(device)

    # Load data
    data = One_RNA_Dataset(root=root)
    train_data, test_data = random_split(data, [0.8, 0.2])

    # Training Data
    train_data_pair = PairDataset(train_data, train_data, sample=True)
    train_loader = DataLoader(
        train_data_pair,
        batch_size=16,
        shuffle=True,
        follow_batch=["x_1", "x_2"],
        num_workers=10)

    # Test Data
    test_data_pair = PairDataset(test_data, test_data, sample=True)
    test_loader = DataLoader(
        test_data_pair,
        batch_size=16,
        shuffle=False,
        follow_batch=["x_1", "x_2"],
        num_workers=10)

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

        # To device
        data_batch = data_batch.to(device)

        # Compute reconstuction loss for graph1
        z1 = model.encode(data_batch.x_1, data_batch.edge_index_1)

        # Compute reconstuction loss for graph2
        z2 = model.encode(data_batch.x_2, data_batch.edge_index_2)

        # Compute loss distance and scalar product between embeddings
        graph1_embedding = torch_geometric.nn.pool.global_mean_pool(z1, data_batch.x_1_batch)
        graph2_embedding = torch_geometric.nn.pool.global_mean_pool(z2, data_batch.x_2_batch)

        distance_predicted = 0
        if distance_from_embedding == "euclidean":
            distance_predicted = torch.norm(graph1_embedding - graph2_embedding, dim=1)  # Euclidean distance between embeddings
        else:
            distance_predicted = (graph1_embedding * graph2_embedding).sum(axis=1)  # Scalar product between embeddings
        
        distances = distance(
            data_batch.edge_index_1,
            data_batch.edge_index_2,
            batch_1=data_batch.x_1_batch,
            batch_2=data_batch.x_2_batch
        )

        if distance_loss == "L1":
            distance_loss = torch.abs(distance_predicted - distances)
        else:
            distance_loss = (distance_predicted - distances)**2

        if distance_loss_only:
            loss = distance_loss.mean()

            loss.backward()
            optimizer.step()

            nb_graphs = data_batch.x_1_batch[-1] + 1
            return loss*nb_graphs, torch.tensor([0], device=device, dtype=torch.float32), loss*nb_graphs

        else:
            loss1 = model.recon_loss(z1, data_batch.edge_index_1)
            loss2 = model.recon_loss(z2, data_batch.edge_index_2)
            loss = (loss1 + loss2) + alpha * distance_loss.mean()

            loss.backward()
            optimizer.step()

            nb_graphs = data_batch.x_1_batch[-1] + 1
            return loss * nb_graphs, (loss1 + loss2) * nb_graphs, distance_loss.sum()

    # Training loop 
    # Loss training
    total_loss_record = []
    total_loss_reconstruction_record = []
    total_loss_distance_record = []

    # Loss eval
    total_loss_test_record = []
    total_loss_reconstruction_test_record = []
    total_loss_distance_test_record = []

    # Best epoch records (for saving the model)
    best_loss = np.inf
    best_epoch = 0

    # Training loop
    for e in tqdm(range(epoch)):
        loss = torch.tensor([0], device=device, dtype=torch.float32)
        loss_reconstruction = torch.tensor([0], device=device, dtype=torch.float32)
        loss_distance = torch.tensor([0], device=device, dtype=torch.float32)
        for batch in train_loader:
            loss_batch, loss_reconstruction_batch, loss_distance_batch = train(batch)
            loss += loss_batch
            loss_reconstruction += loss_reconstruction_batch
            loss_distance += loss_distance_batch
        total_loss_record.append(loss.item()/len(train_data))
        total_loss_reconstruction_record.append(loss_reconstruction.item()/len(train_data))
        total_loss_distance_record.append(loss_distance.item()/len(train_data))

        if e % (max(1, epoch//20)) == 0:
            # Evaluate model on test data
            loss_test = 0
            loss_reconstruction_test = 0
            loss_distance_test = 0
            for batch in test_loader:
                loss_batch, loss_reconstruction_batch, loss_distance_batch = train(batch)
                loss_test += loss_batch
                loss_reconstruction_test += loss_reconstruction_batch
                loss_distance_test += loss_distance_batch
            total_loss_test_record.append(loss_test.item()/len(test_data))
            total_loss_reconstruction_test_record.append(loss_reconstruction_test.item()/len(test_data))
            total_loss_distance_test_record.append(loss_distance_test.item()/len(test_data))

            # Save model if it is the best one
            if loss_test < best_loss:
                best_loss = loss_test
                best_epoch = e
                torch.save(model.state_dict(), os.path.join("models", save_folder, "best_model.pt"))
    
    # Save last model
    name_model = f"alpha_{alpha}_hidden_size_{model.encoder.hidden_channels}_epoch_{epoch}_distance_loss_only_{distance_loss_only}_last_epoch"
    torch.save(model, os.path.join("models", save_folder, name_model + ".pt"))


    logging.info(f"Total loss {total_loss_record[-1]}")
    logging.info(f"Reconstruction loss {total_loss_reconstruction_record[-1]}")
    logging.info(f"Distance loss {total_loss_distance_record[-1]}")

    loss_training = [total_loss_record, total_loss_reconstruction_record, total_loss_distance_record]
    loss_eval = [total_loss_test_record, total_loss_reconstruction_test_record, total_loss_distance_test_record]
    return model, loss_training, loss_eval


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

