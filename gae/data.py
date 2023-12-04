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

        structures = self.generate_suboptimal_structures(self.raw_paths[0], n=self.n)
        data_list = [self.structure_to_data(structure, sequence) for structure in structures]
        print(f"Number of structures : {len(data_list)}")

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
            x_1=x_1, edge_index_1=edge_index_1, edge_attr_1=edge_attr_1, x_2=x_2, edge_index_2=edge_index_2, edge_attr_2=edge_attr_2, **kwargs
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
    """
    def __init__(self, dataset_1, dataset_2, sample=False):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.sample = sample

    def __len__(self):
        return len(self.dataset_1) if self.sample else len(
            self.dataset_1) * len(self.dataset_2)

    def __getitem__(self, idx):
        if self.sample:
            data_1 = self.dataset_1[idx]
            data_2 = self.dataset_2[random.randint(0, len(self.dataset_2) - 1)]
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