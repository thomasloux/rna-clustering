# file to allow quick test of the model 
# use arg parser to allow for different parameters from the command line

from gae import get_couple_trained_model, get_vanilla_model
from utils import base_pair_distance

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import logging

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
parser.add_argument("--distance_loss_only",
                    type=bool,
                    default=False,
                    help="Only use distance loss for training")

#########

args = parser.parse_args()
epoch = args.epoch
alpha = args.alpha
hidden_size = args.hidden_size
name = args.name
device_name = args.device
distance_loss_only = args.distance_loss_only

# Check if model already exists
if os.path.exists(os.path.join("models", name)) and not os.path.isdir(os.path.join("models", name)):
    raise ValueError("Model already exists")
os.mkdir(os.path.join("models", name))

print(f"Training : alpha: {alpha}, size: {hidden_size}")
model = get_vanilla_model(hidden_channels=hidden_size)
logging.basicConfig(filename=os.path.join("models", name, "log.txt"), level=logging.INFO)
try:
    model, losses_liste_training, losses_liste_eval = get_couple_trained_model(
        epoch=epoch,
        model=model,
        distance=base_pair_distance,
        save_folder=name,
        alpha=alpha,
        device=device_name,
        distance_from_embedding="euclidean",
        distance_loss_only=distance_loss_only)
except Exception as e:
    logging.exception(e)
    raise e

def save_losses(losses_liste, name_loss):
    names = ["total_loss_record", "total_loss_reconstruction_record", "total_loss_distance_record"]
    #total_loss_record, total_loss_reconstruction_record, total_loss_distance_record = losses_liste

    for i in range(len(losses_liste)):
        plt.plot(losses_liste[i], label=names[i])
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        #plt.yscale('log')
        plt.legend()
    plt.savefig(os.path.join("models", name, name_loss + " figure.png"))
    plt.close()

save_losses(losses_liste_training, "training")
save_losses(losses_liste_eval, "eval")


