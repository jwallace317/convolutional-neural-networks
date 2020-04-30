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

    # instantiate train data loader
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=1000)

    # instantiate test data loader
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

    # instantiate base architecture convolutional neural network
    base_cnn = BaseCNN()

    # train the base architecture convolutional neural network
    base_cnn.train(train_data_loader, learning_rate=0.01, n_epochs=10)

    # test the base architecture convolutional neural network
    base_cnn.test(test_data_loader)


if __name__ == '__main__':
    main()
