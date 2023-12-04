# file to allow quick test of the model 
# use arg parser to allow for different parameters from the command line

from gae import get_couple_trained_model, base_pair_distance, get_vanilla_model

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
sns.set_theme(style="darkgrid")

# Parse argument
parser = argparse.ArgumentParser()

parser.add_argument("--epoch",
                    type=int,
                    default=10,
                    help="Number of epoch to train")
parser.add_argument("--alpha",
                    type=float,
                    default=1.0,
                    help="Weight parameter to balance distance and reconstruction loss")
parser.add_argument("--hidden-size",
                    type=int,
                    default=32,
                    help="Size of the hidden vector")
#########

args = parser.parse_args()
epoch = args.epoch
alpha = args.alpha
hidden_size = args.hidden_size

print(f"Training : alpha: {alpha}, size: {hidden_size}")
model = get_vanilla_model(hidden_channels=hidden_size)
model, losses_liste = get_couple_trained_model(epoch=epoch, model=model, distance=base_pair_distance, alpha=alpha)

names = ["total_loss_record", "total_loss_reconstruction_record", "total_loss_distance_record"]
#total_loss_record, total_loss_reconstruction_record, total_loss_distance_record = losses_liste

path = "gae/models/alphas_hidden_size_test/"

name = f"alpha_{alpha}_hidden_size_{hidden_size}_epoch_{epoch}"

torch.save(model, os.path.join(path, name))

for i in range(len(losses_liste)):
    plt.plot(losses_liste[i], label=names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    #plt.yscale('log')
    plt.legend()
plt.savefig("gae/figures/" + name + ".png")
plt.close()


