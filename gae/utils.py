import numpy as np
import torch
from torch import Tensor
from typing import Optional, Union


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
        return torch.tensor(liste)
    else:
        edges1 = set(map(tuple, structure_1.t().detach().numpy()))
        edges2 = set(map(tuple, structure_2.t().detach().numpy()))
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
