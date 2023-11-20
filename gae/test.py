## Test model training 

from gae import get_couple_trained_model, base_pair_distance, get_vanilla_model

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sns.set_theme(style="darkgrid")


def test():
    epoch = 3
    alphas = [0, 1, 10, 50]

    hidden_channel_size = [32]

    for alpha in alphas:
        for size in hidden_channel_size:
            print(f"Training : alpha: {alpha}, size: {size}")
            model = get_vanilla_model(hidden_channels=size)
            model, losses_liste = get_couple_trained_model(epoch=epoch, model=model, distance=base_pair_distance, alpha=alpha)

            names = ["total_loss_record", "total_loss_reconstruction_record", "total_loss_distance_record"]
            #total_loss_record, total_loss_reconstruction_record, total_loss_distance_record = losses_liste
            
            path = "gae/models/alphas_hidden_size_test/"

            name = f"alpha_{alpha}_hidden_size_{size}_epoch_{epoch}"

            torch.save(model, os.path.join(path, name))

            for i in range(len(losses_liste)):
                plt.plot(losses_liste[i], label=names[i])
                plt.xlabel('Epoch')
                plt.ylabel('Train loss')
                #plt.yscale('log')
                plt.legend()
            plt.savefig("gae/figures/" + name + ".png")
            plt.close()

if __name__ == "__main__":
    test()






