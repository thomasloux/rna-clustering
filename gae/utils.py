import numpy as np
import torch
from torch import Tensor
from typing import Optional, Union
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.nn import GAE
from torch_geometric.nn.pool import global_mean_pool
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import PairDataset


def base_pair_distance(structure_1: Tensor, structure_2: Tensor,
        batch_1: Optional[Tensor] = None, batch_2: Optional[Tensor] = None) -> Union[float, Tensor]:
    """
    Compute the base pair distance between two RNA structures
    it corresponds to the symetric difference between the two sets of base pairs

    :param structure1: first RNA structure
    :param structure2: second RNA structure
    :param batch_1: batch index for graph 1, format [sum(num_nodes) in the batch] like [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    :param batch_2: batch index for graph 2
    :return: base pair distance
    """
    if batch_1 is not None and batch_2 is not None:
        # Compute the base pair distance for each graph in the batch
        liste = []
        for i in range(batch_1[-1] + 1):
            node_range_1 = torch.argwhere(batch_1 == i).flatten()
            edges_1 = structure_1[:, torch.isin(structure_1[0, :], node_range_1)]
            node_range_2 = torch.argwhere(batch_2 == i).flatten()
            edges_2 = structure_2[:, torch.isin(structure_2[0, :], node_range_2)]
            liste.append(base_pair_distance(edges_1, edges_2))
        return torch.tensor(liste, dtype=torch.int16, device=structure_1.device)
    else:
        edges1 = set(map(tuple, structure_1.t().cpu().detach().numpy()))
        edges2 = set(map(tuple, structure_2.t().cpu().detach().numpy()))
        # Divide by 2 because the symetric difference counts each edge twice
        distance = float(len(edges1.symmetric_difference(edges2)))/2
        return distance

def base_pair_distance_renorm(structure_1: Tensor, structure_2: Tensor,
        batch_1: Optional[Tensor] = None, batch_2: Optional[Tensor] = None) -> Union[float, Tensor]:
    """
    Compute the base pair distance between two RNA structures
    it corresponds to the symetric difference between the two sets of base pairs

    :param structure1: first RNA structure
    :param structure2: second RNA structure
    :param batch_1: batch index for graph 1, format [sum(num_nodes) in the batch] like [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    :param batch_2: batch index for graph 2
    :return: base pair distance

    Note : you have to provide the batch index for the two graphs 
    even for a single graph
    """
    if batch_1 is not None and batch_2 is not None:
        # Compute the base pair distance for each graph in the batch
        liste = []
        for i in range(batch_1[-1] + 1):
            # Extract nodes id for graph
            # and edges for the i-th graph
            node_range_1 = torch.argwhere(batch_1 == i).flatten()
            edges_1 = structure_1[:, torch.isin(structure_1[0, :], node_range_1)]
            node_range_2 = torch.argwhere(batch_2 == i).flatten()
            edges_2 = structure_2[:, torch.isin(structure_2[0, :], node_range_2)]

            # Compute the base pair distance for the i-th graph
            distance = base_pair_distance(edges_1, edges_2)
            # Renormalize the distance by the number of nodes in the two graphs
            distance = distance / (len(node_range_1) + len(node_range_2))

            liste.append(distance)
        return torch.tensor(liste)
    else:
        edges1 = set(map(tuple, structure_1.t().detach().numpy()))
        edges2 = set(map(tuple, structure_2.t().detach().numpy()))
        # Divide by 2 because the symetric difference counts each edge twice
        distance = float(len(edges1.symmetric_difference(edges2)))/2
        return distance

# Make the graph of real distance vs predicted distance
def plot_correlation_prediction(model: GAE, dataset: PairDataset, title: str):
    """
    Plot the correlation between the real distance and the predicted distance.

    :param dataset: dataset to use (Must provide PairDataset)
    :param title: title of the plot (train or test)
    """
    real_distances = []
    predicted_distances = []

    # Prepare the data loader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        follow_batch=["x_1", "x_2"],
        num_workers=20)

    for data in tqdm(loader):
        # Compute the real distance
        real_distances.append(
            base_pair_distance(
                data.edge_index_1,
                data.edge_index_2,
                data.x_1_batch,
                data.x_2_batch
            )
        )
        # Compute the predicted distance
        with torch.no_grad():
            z1 = model.encode(data.x_1, data.edge_index_1)
            z2 = model.encode(data.x_2, data.edge_index_2)

            graph1_embedding = global_mean_pool(z1, data.x_1_batch)
            graph2_embedding = global_mean_pool(z2, data.x_2_batch)

            predicted_distances.append(
                torch.norm(graph1_embedding - graph2_embedding, dim=1)
            )

    real_distances = torch.cat(real_distances).detach().numpy()
    predicted_distances = torch.cat(predicted_distances).detach().numpy()

    fig = plt.figure()
    plt.scatter(real_distances, predicted_distances)
    plt.xlabel("Real distance")
    plt.ylabel("Predicted distance")
    plt.title("Real distance vs predicted distance " + title)
    # Plot the identity line
    plt.plot([0, np.max(real_distances)], [0, np.max(real_distances)], c='black', linestyle='--')
    return fig, real_distances, predicted_distances
