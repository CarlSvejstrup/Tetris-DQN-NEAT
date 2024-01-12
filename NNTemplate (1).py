import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy
import matplotlib.pyplot as plt
import os


class MLP(nn.Module):
    """
    A generic Neural Network to predict an output from an input
    to be modified based on your specific use case.
    For classification use an appropriate loss function such as
    Cross-Entropy
    """
    def __init__(self,
                 inputSize,
                 outputSize,
                 nHiddenLayers=6,
                 nNodes=128,
                 lr=2e-4,
                 dtype=torch.float32,
                 activationFunction=nn.ReLU(),
                 optimiser=torch.optim.Adam,
                 lossFunction=nn.MSELoss(),
                 constrainZeroOne=False,):
        super().__init__()
        self.layers = nn.ModuleList()
        self.outputSize = outputSize
        self.mse_loss = lossFunction

        for k in range(nHiddenLayers):
            newSize = nNodes  # int(nNodes - nNodes * (k / (nHiddenLayers * 2))) for decreasing number of nodes per layer
            self.layers.append(nn.Linear(inputSize, newSize, dtype=dtype))
            self.layers.append(activationFunction)
            inputSize = copy(newSize)

        self.layers.append(nn.Linear(inputSize, outputSize, dtype=dtype))
        if constrainZeroOne:
            self.layers.append(nn.Sigmoid())    # To constrain output between (0, 1)

        self.optimiser = optimiser(self.parameters(), lr=lr)    # I do not apologise for the British spelling of optimiser

        # self.scheduler = ReduceLROnPlateau(self.optimiser, 'min') Look into scheduling if interested

    # Define the forward function of the neural network
    def forward(self, X):
        x = X
        for layer in self.layers:
            x = layer(x)
        return x

    def trainBatch(self, X_batch, targets, epochs=1):
        pbar = tqdm(range(epochs), desc=f'Training')
        targets = targets.reshape((-1, self.outputSize))
        losses = torch.zeros(epochs)

        for i in pbar:
            outputs = self.forward(X_batch)

            loss = self.mse_loss(outputs, targets)
            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()
            pbar.set_description(f'Training Error: {loss}')
            losses[i] = loss
            # self.scheduler.step(loss)
        return losses

        # return losses

    def error(self, X_batch, targets):
        outputs = self.forward(X_batch)
        return self.mse_loss(outputs, targets.reshape((-1, 1)))

    def save(self, filename):
        torch.save(self.state_dict(), filename + '.pt')

    def load(self, filename):
        self.load_state_dict(torch.load(filename + '.pt'))


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        print('Using Cuda')
    else:
        print('Cuda not available')

    # Hyper-parameters:
    nNodes = 300
    nHiddenLayers = 4
    epochs = 1000

    dtype = torch.float32

    X_train_numpy = np.random.standard_normal((100, 1))
    X_test_numpy = np.random.standard_normal((100, 1))
    y_train_numpy = np.random.standard_normal((100, 1))
    y_test_numpy = np.random.standard_normal((100, 1))

    X_train, X_test = torch.tensor(X_train_numpy, dtype=dtype), torch.tensor(X_test_numpy, dtype=dtype)
    y_train, y_test = torch.tensor(y_train_numpy, dtype=dtype), torch.tensor(y_test_numpy, dtype=dtype)

    # Create and train network
    network = MLP(
        inputSize=X_train.size(1),
        outputSize=y_train.size(1),
        nHiddenLayers=nHiddenLayers,
        nNodes=nNodes,
        optimiser=torch.optim.Adam,
        activationFunction=nn.ReLU(),
    )
    # Here the network is trained:
    network.trainBatch(X_train, y_train, epochs=epochs)

    if not os.path.isdir(os.path.join(os.getcwd(), 'Models')):
        os.makedirs(os.path.join(os.getcwd(), 'Models'))
    if not os.path.isdir(os.path.join(os.getcwd(), 'Plots')):
        os.makedirs(os.path.join(os.getcwd(), 'Plots'))
    
    # Then saved, here the hyperparameters are used in the filename to distinguish them:
    network.save(f'Models/NN_{nNodes}_{nHiddenLayers}_{epochs}')

    # To load again use
    network.load(f'Models/NN_{nNodes}_{nHiddenLayers}_{epochs}')

    # For analysis of your results, use the following to get the NN prediction on the test data:
    y_NN = network.forward(X_test)

    # Here the difference for each point is plotted, and the mean error is displayed:
    y_NN_plot = y_NN.cpu().detach().numpy()
    y_test_plot = y_test_numpy
    DiffSeries = y_test_plot - y_NN_plot
    ME = np.abs(DiffSeries).mean()

    alpha = .7  # Transparency of the points, so we can see overlapping points
    plt.scatter(range(len(DiffSeries)), DiffSeries,
                label='Error', alpha=alpha, s=.7,
                facecolors='none', edgecolors='b', )
    plt.title(f'Error between Dref and NN demand over time ME={ME}')
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Plots/DifferencePlot_{nNodes}_{nHiddenLayers}_{epochs}.png')
    plt.show()

