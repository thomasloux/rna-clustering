# file to allow quick test of the model 
# use arg parser to allow for different parameters from the command line

from gae import get_couple_trained_model, get_vanilla_model, 
from utils import base_pair_distance

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

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
parser.add_argument("--name",
                    type=str,
                    default="default",
                    help="Name of the model and files")
parser.add_argument("--device",
                    type=str,
                    default="cpu",
                    help="Device to use for training")

#########

args = parser.parse_args()
epoch = args.epoch
alpha = args.alpha
hidden_size = args.hidden_size
name = args.name
device_name = args.device

# Check if model already exists
if os.path.exists(os.path.join("models", name)):
    raise ValueError("Model already exists")

print(f"Training : alpha: {alpha}, size: {hidden_size}")
model = get_vanilla_model(hidden_channels=hidden_size)
model, losses_liste = get_couple_trained_model(epoch=epoch, model=model, distance=base_pair_distance, alpha=alpha)

names = ["total_loss_record", "total_loss_reconstruction_record", "total_loss_distance_record"]
#total_loss_record, total_loss_reconstruction_record, total_loss_distance_record = losses_liste

os.mkdir(os.path.join("models", name))
name_model = f"alpha_{alpha}_hidden_size_{hidden_size}_epoch_{epoch}"
torch.save(model, os.path.join("models", name, name_model + ".pt"))

for i in range(len(losses_liste)):
    plt.plot(losses_liste[i], label=names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    #plt.yscale('log')
    plt.legend()
plt.savefig(os.path.join("models", name, name_model + ".png"))
plt.close()


