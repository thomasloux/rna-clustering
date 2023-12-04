# rna-clustering
Using GNN to cluster non coding RNA

## Run training
While being in ./rna-clustering
python gae/train.py --epoch 30 --alpha 0.1 --hidden-size 32 --name test_eval --device cuda --distance_loss_on
ly True
See train.py for more details about the arguments

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