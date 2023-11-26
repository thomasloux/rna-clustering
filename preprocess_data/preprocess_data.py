# Transform sequences to secondary structure sequences

import os
import sys
import numpy as np
import RNA
from Bio import SeqIO
import pandas as pd


def fasta_to_pandas_point_bracket(fasta_file: str, point_bracket_file: str):
    """
    Convert fasta file to point bracket file
    """
    seqs = SeqIO.parse(fasta_file, 'fasta')
    df = pd.DataFrame(columns=['id', 'seq', 'struct'])
    for i, seq in enumerate(seqs):
        seq = str(seq.seq)
        struct = RNA.fold(seq)[0]
        df.loc[i] = [i, seq, struct]
    df.to_csv(point_bracket_file, index=False)

def point_bracket_to_adjacency_mathrix(secStructure: str):
    """
    Convert secondary structure to adjacency matrix
    """
    n = len(secStructure)
    adj = np.zeros((n, n))
    stack = []
    for i, s in enumerate(secStructure):
        if s == '(':
            stack.append(i)
        elif s == ')':
            j = stack.pop()
            adj[i, j] = 1
            adj[j, i] = 1
    return adj
