"""
Task Main
"""

# import necessary modules
import numpy as np
import torch
import torch.nn.functional as func
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils
from convolutional_neural_networks import BaseCNN
from multilayer_perceptron import MLP


def main():
    """
    Task Main
    """

    # download and parse Fashion MNIST training set
    train_set = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    # download and parse Fashion MNIST testing set
    test_set = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    # instantiate multilayer perceptron
    mlp = MLP()

    # instantiate batch loader
    loader = torch.utils.data.DataLoader(train_set, batch_size=10000)

    # train the multilayer perceptron
    mlp.train(loader, learning_rate=0.01, n_epochs=10)


if __name__ == '__main__':
    main()
