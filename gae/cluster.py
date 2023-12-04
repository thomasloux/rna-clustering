import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.multiprocessing import Pool, set_start_method
import numpy as np
from tqdm import tqdm
from utils import base_pair_distance
from data import PairDataset, One_RNA_Dataset

def dissimilarity_matrix(dataset: Dataset) -> np.ndarray:
    """
    Compute the dissimilarity matrix of a dataset
    Attention: this function is very slow, use it only for small dataset

    Args:
        dataset (Dataset): dataset of RNA

    Returns:
        np.ndarray: dissimilarity matrix
    """
    X = np.zeros((len(dataset), len(dataset))).flatten()
    pair_dataset = PairDataset(dataset, dataset, sample=False)

    batch_size = 100
    device = torch.device('cuda')

    data_loader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=False, follow_batch=["x_1", "x_2"], num_workers=5)

    distance_liste = []

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            distances = base_pair_distance(
                data.edge_index_1.to(device),
                data.edge_index_2.to(device),
                batch_1=data.x_1_batch.to(device),
                batch_2=data.x_2_batch.to(device))
            distance_liste.append(distances.cpu().detach())

    X = np.concat(distance_liste)
    X = X.reshape((len(dataset), len(dataset)))
    return X
    