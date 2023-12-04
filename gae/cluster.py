import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.multiprocessing import Pool, set_start_method
import numpy as np
from tqdm import tqdm
from utils import base_pair_distance
from data import PairDataset, One_RNA_Dataset
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import KMeans

def dissimilarity_matrix(dataset: Dataset) -> np.ndarray:
    """
    Compute the dissimilarity matrix of a dataset
    Attention: this function is very slow, use it only for small dataset

    Remarks: It is faster on cuda but still ok on cpu. Must choose wisely the batch size
    and num_workers.

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
    
def compute_K_means_model(
    dataset: Dataset,
    model: str,
    device: str = 'cpu',
    n_clusters: int = 2,
    **kwargs):
    """
    Compute K-means clustering on a dataset

    Args:
        dataset (Dataset): dataset of RNA
        model (str): Path to model to use for clustering
        device (str, optional): Device to use. Defaults to 'cpu'.
        n_clusters (int, optional): Number of clusters. Defaults to 2.
        **kwargs: kwargs for KMeans

    Returns:
        np.ndarray: cluster labels
    """
    model = get_vanilla_model(32)
    model.load_state_dict(torch.load(model))

    device = torch.device(device)
    model.to(device)

    df = pd.DataFrame(columns=['id', 'pred_embedding'])
    for i in range(len(dataset)):
        data = dataset[i]
        data.x.to(device)
        data.edge_index.to(device)
        with torch.no_grad():
            pred = model.encode(data.x, data.edge_index)
        df.loc[i] = [i, pred.cpu().numpy()]

    X = np.array(df['pred_embedding'].tolist())
    X = np.mean(X, axis=1)

    clustering = KMeans(n_clusters=n_clusters, **kwargs)
    clustering.fit(X)
    return clustering

def compute_HCA_model(
    dataset: Dataset,
    model: str,
    device: str = 'cpu',
    **kwargs):
    """
    Compute K-means clustering on a dataset

    Args:
        dataset (Dataset): dataset of RNA
        model (str): Path to model to use for clustering
        device (str, optional): Device to use. Defaults to 'cpu'.
        **kwargs: kwargs for AgglomerativeClustering


    Returns:
        np.ndarray: cluster labels
    """
    model = get_vanilla_model(32)
    model.load_state_dict(torch.load(model))

    device = torch.device(device)
    model.to(device)

    df = pd.DataFrame(columns=['id', 'pred_embedding'])
    for i in range(len(dataset)):
        data = dataset[i]
        data.x.to(device)
        data.edge_index.to(device)
        with torch.no_grad():
            pred = model.encode(data.x, data.edge_index)
        df.loc[i] = [i, pred.cpu().numpy()]

    X = np.array(df['pred_embedding'].tolist())
    X = np.mean(X, axis=1)

    hca_clustering = AgglomerativeClustering(affinity='euclidean', **kwargs)
    hca_clustering.fit(X)
    return hca_clustering

def plot_dendrogram(
    model,
    filename: str = 'dendrogram.png',
    **kwargs):
    """
    Plot dendrogram from sklearn model

    Example: plot_dendrogram(hca_clustering, truncate_mode='level', p=10)
    Can't only be used with AgglomerativeClustering for distance_threshold=0

    Args:
        model (sklearn model): model to plot
        **kwargs: kwargs for dendrogram
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(filename)
