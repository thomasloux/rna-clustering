# file to allow quick test of the model 
# use arg parser to allow for different parameters from the command line

from gae import get_couple_trained_model, get_vanilla_model
from utils import base_pair_distance, plot_correlation_prediction
from data import One_RNA_Dataset, PairDataset

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
import argparse
import logging
from utils import base_pair_distance
from torch_geometric.nn import conv
from torch_geometric.data import DataLoader


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
parser.add_argument("--distance",
                    type=str,
                    default="euclidean",
                    help="Distance to use for training")
parser.add_argument("--distance_loss",
                    type=str,
                    default="L2",
                    help="Loss to compare distance and predicted distance")
parser.add_argument("--distance_loss_only",
                    type=bool,
                    default=False,
                    help="Only use distance loss for training")
parser.add_argument("--layer",
                    type=str,
                    default="GCNConv",
                    help="Type of Message Passing layer to use")

#########

args = parser.parse_args()
epoch = args.epoch
alpha = args.alpha
hidden_size = args.hidden_size
name = args.name
device_name = args.device
distance = args.distance
distance_loss = args.distance_loss
distance_loss_only = args.distance_loss_only
layer = args.layer

# Check if model already exists
if os.path.exists(os.path.join("models", name)) and not os.path.isdir(os.path.join("models", name)):
    raise ValueError("Model already exists")
os.mkdir(os.path.join("models", name))

logging.basicConfig(filename=os.path.join("models", name, "log.txt"), level=logging.INFO)

# Write the parameters in a file
with open(os.path.join("models", name, "parameters.txt"), "w") as f:
    f.write(f"Epoch: {epoch}\n")
    f.write(f"Alpha: {alpha}\n")
    f.write(f"Hidden size: {hidden_size}\n")
    f.write(f"Device: {device_name}\n")
    f.write(f"Distance: {distance}\n")
    f.write(f"Distance loss: {distance_loss}\n")
    f.write(f"Distance loss only: {distance_loss_only}\n")
    f.write(f"Layer: {layer}\n")

print(f"Training : alpha: {alpha}, size: {hidden_size}")
try:
    layer = getattr(conv, layer)
except Exception as e:
    logging.exception(e)
    raise e
model = get_vanilla_model(hidden_channels=hidden_size, layer=layer)

# Load data
data = One_RNA_Dataset(root="data/test")
train_data, test_data = random_split(data, [0.8, 0.2])

try:
    model, losses_liste_training, losses_liste_eval = get_couple_trained_model(
        epoch=epoch,
        model=model,
        distance=base_pair_distance,
        train_data=train_data,
        test_data=test_data,
        save_folder=name,
        alpha=alpha,
        device=device_name,
        distance_from_embedding="euclidean",
        distance_loss_only=distance_loss_only)
except Exception as e:
    logging.exception(e)
    raise e

### Plot the losses
names = ["total_loss_record", "total_loss_reconstruction_record", "total_loss_distance_record"]
    #total_loss_record, total_loss_reconstruction_record, total_loss_distance_record = losses_liste

fig, ax = plt.subplots(1, 2, figsize=(20, 5))

losses_liste = [losses_liste_training, losses_liste_eval]
names_plot = ["Training", "Evaluation"]

for j in range(len(losses_liste)):
    for i in range(len(losses_liste[j])):
        ax[j].plot(losses_liste[j][i], label=names[i])
    ax[j].set_xlabel('Epoch')
    ax[j].set_ylabel('Loss')
    ax[j].set_title(f"{names_plot[j]} losses")
        # add legend
        #plt.yscale('log')
    ax[j].legend()

plt.savefig(os.path.join("models", name, "fig.png"))
plt.close()

###
model.to("cpu")
# Plot the correlation between the real distance and the predicted distance
fig_train, _, _ = plot_correlation_prediction(model, train_data, "train")
fig_test, _, _ = plot_correlation_prediction(model, test_data, "test")

fig_train.savefig(os.path.join("models", name, "correlation_train.png"))
fig_test.savefig(os.path.join("models", name, "correlation_test.png"))



