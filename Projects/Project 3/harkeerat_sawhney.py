'''
Generation of news titles using LSTM
Data taken from: https://www.kaggle.com/datasets/rmisra/news-category-dataset/
Student: Harkeerat Singh Sawhney
'''
# Packages
import pandas as pd
import pickle
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


def train_with_TBBTT(max_epochs, model, dataloader, criterion, optimizer, chunk_size, device, clip=None):
    losses = []
    perplexities = []
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        model.train()
        for input, output in dataloader:
            # Get the number of chunks
            n_chunks = input.shape[1] // chunk_size

            # Loop on chunks
            for j in range(n_chunks):
                # TODO what is missing here?
                # Switch between the chunks
                if j < n_chunks - 1:
                    input_chunk = input[:, j * chunk_size:(j + 1) * chunk_size].to(device).to(torch.int64)
                    output_chunk = output[:, j * chunk_size:(j + 1) * chunk_size].to(device).to(torch.int64)
                else:
                    input_chunk = input[:, j * chunk_size:].to(device).to(torch.int64)
                    output_chunk = output[:, j * chunk_size:].to(device).to(torch.int64)
                # Initialise model's state and perform forward pass
                # If it is the first chunk, initialise the state to 0
                if j == 0:
                    h, c = # TODO ?
                else:  # Initialize the state to the previous state - detached!
                    h, c = # TODO ?

                # Forward step
                # TODO: complete the forward step

                # Calculate loss
                # TODO complete the loss calculation

                # Calculate gradients and update parameters
                # TODO: complete the backward step
                # Clipping if needed
                # TODO: complete the clipping step
                # Update parameters
                # TODO: complete the update step

        # Print loss and perplexity every epoch
        # TODO: complete the perplexity calculation and loss / perpl priting

        # TODO keep track of losses and perplexities
        losses.append(0)
        perplexities.append(0) # <--- Replace here

        model.eval()
        # TODO prompt a sentence from the model

    return model, losses, perplexities

if __name__ == "__main__":
    print("Do the exercise here!")