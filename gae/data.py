# Data class 
# Purpose : define classes to take into account that the unit data is a couple of graphs
# Necessity : Define a Data Loader to merge couples of graphes into batches
# Inspired by : deep-graph-matching-consensus

import torch
import re
from torch_geometric.data import Data, Dataset
import random
import subprocess
import os.path as osp
import torch.nn.functional as F
import numpy as np
import pandas as pd

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
    n = 1000

    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        """
        :param root: root directory where the dataset should be saved.
            This folder is split into raw_dir (downloaded dataset) and processed_dir (processed dataset).
        :param n: number of suboptimal structures to generate
        """
        self.root = root
        super(One_RNA_Dataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['test.seq']

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.n)]

    def download(self):
        raise NotImplementedError('Raw data not found.')

    def structure_to_data(self, structure: str, sequence: str) -> Data:
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
        # Hot one encoding of the edge type
        edge_attributes = torch.cat(
            [
                torch.tensor([[1, 0]]).repeat(edge_index_strong.size(1)*2, 1),
                torch.tensor([[0, 1]]).repeat(edge_index_weak.size(1)*2, 1)
            ]
        )
         
        data = Data(x=x, edge_index=edge_pairs, edge_attr=edge_attributes, num_nodes=len(sequence))

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

        ## Use a sample of the suboptimal structures
        command = subprocess.run([
            "RNAsubopt",
            "--stochBT_en", str(n),
            "-i", input_file,
            "--sorted",
            "--nonRedundant",
            "-T", str(80)], capture_output=True)

        ## Use all the suboptimal structures with a max energy from optimal structure
        # command = subprocess.run["RNAsubopt", "-e", str(n), "-i", input_file]

        output = command.stdout.decode('utf-8').split('\n')
        # Remove first line (sequence) and last line (empty)
        output = output[1:-1]
        return output

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_paths[0], 'r') as f:
            sequence = f.readline().strip()

        structures = self.generate_suboptimal_structures(self.raw_paths[0], n=self.n)

        # Clean structures
        structures = [x.split() for x in structures]
        structures = [x for x in structures if x] # Remove empty lines

        # Sort structures by probability (most probable first)
        df = pd.DataFrame(structures, columns=['structure', 'energy', 'probability'])
        df.sort_values(by=['probability'], inplace=True)

        # Save the dataframe
        df.to_csv(osp.join(self.root, 'structures.csv'))

        structures = df['structure'].values

        data_list = [self.structure_to_data(structure, sequence) for structure in structures]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        for i, data in enumerate(data_list):
            torch.save(data, osp.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

class PairData(Data):
    """
    Class to handle a pair of graphs
    """
    def __init__(self, x_1=None, edge_index_1=None, edge_attr_1=None, x_2=None, edge_index_2=None, edge_attr_2=None, **kwargs):
        """
        Enforce the necessary arguments to be PairData objects
        """
        super(PairData, self).__init__(
            x_1=x_1,
            edge_index_1=edge_index_1,
            edge_attr_1=edge_attr_1,
            x_2=x_2,
            edge_index_2=edge_index_2,
            edge_attr_2=edge_attr_2,
            **kwargs
            )
        
    def __inc__(self, key, value, *args):
        if bool(re.search('index_1', key)):
            return self.x_1.size(0)
        if bool(re.search('index_2', key)):
            return self.x_2.size(0)
        else:
            return 0

class PairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building pairs between separate dataset examples.

    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
        remove_random (bool, optional): If set to :obj:`True`, will remove
            random target examples for every source example. (default: :obj:`False`)
            If True, the pairs will always be the same
            Use False for training and True for testing between multiple models
    """
    def __init__(self, dataset_1, dataset_2, sample=False, remove_random=False):
        if not sample and remove_random:
            raise ValueError("Cannot remove random if sample is False")
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.sample = sample
        self.remove_random = remove_random
        self.rng = np.random.default_rng(seed=42)
        if self.remove_random:
            self.random_index = self.rng.integers(0, len(self.dataset_2), size=len(self.dataset_1))

    def __len__(self):
        return len(self.dataset_1) if self.sample else len(self.dataset_1) * len(self.dataset_2)

    def __getitem__(self, idx):
        if self.sample:
            if self.remove_random:
                data_2 = self.dataset_2[self.random_index[idx]]
            else:
                data_2 = self.dataset_2[self.rng.integers(0, len(self.dataset_2))]
            data_1 = self.dataset_1[idx]
        else:
            data_1 = self.dataset_1[idx // len(self.dataset_2)]
            data_2 = self.dataset_2[idx % len(self.dataset_2)]

        return PairData(
            x_1=data_1.x,
            edge_index_1=data_1.edge_index,
            edge_attr_1=data_1.edge_attr,
            x_2=data_2.x,
            edge_index_2=data_2.edge_index,
            edge_attr_2=data_2.edge_attr,
            num_nodes=None,
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)
