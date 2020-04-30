"""
Task Main
"""

# import necessary modules
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from convolutional_neural_networks import BaseCNN


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

    # instantiate base convolutional neural network
    base_cnn = BaseCNN()

    # instantiate batch loader
    loader = torch.utils.data.DataLoader(train_set, batch_size=1)

    # instantiate optimizer
    optimizer = optim.Adam(base_cnn.parameters(), lr=0.001)


if __name__ == '__main__':
    main()
