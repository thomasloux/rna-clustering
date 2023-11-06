# rna-clustering
Using GNN to cluster non coding RNA

## Useful functions
RNA (from ViennaRNA)
- `fold` : RNA secondary structure prediction
- `eval_structure` : RNA secondary structure evaluation

Using CLI for ViennaRNA
RNAsubopt --stochBT_en=3 -s < test.seq
-> Permet d'avoir un tirage de structure secondaire
RNAsubopt -e 0.1 -s < test.seq
-> Permet d'avoir la liste des structures secondaires avec une énergie max de 0.1 kcal/mol par rapport à la structure optimale

## Useful links
- Varna : https://gitlab.inria.fr/amibio/varna-api


## Dependencies :
- ViennaRNA
- Biopython
- Varna