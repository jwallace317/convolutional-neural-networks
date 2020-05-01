"""
Task Main
"""

# import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as func
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils
from convolutional_neural_networks import CNN1, CNN2, BaseCNN
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

    # instantiate multilayer perceptron neural network
    mlp = MLP()

    # train the multilayer perceptron neural network
    mlp_performance = mlp.train(
        train_data_loader, learning_rate=0.01, n_epochs=20)

    # test the multilayer perceptron neural network
    mlp.test(test_data_loader)

    # instantiate base architecture convolutional neural network
    base_cnn = BaseCNN()

    # train the base architecture convolutional neural network
    base_cnn_performance = base_cnn.train(
        train_data_loader, learning_rate=0.01, n_epochs=20)

    # test the base architecture convolutional neural network
    base_cnn.test(test_data_loader)

    # instantiate the variant 1 architecture convolutional neural network
    cnn1 = CNN1()

    # train the variant 1 architecture convolutional neural network
    cnn1_performance = cnn1.train(
        train_data_loader, learning_rate=0.01, n_epochs=20)

    # test the variant 1 architecture convolutional neural  network
    cnn1.test(test_data_loader)

    # instantiate the variant 2 architecture convolutional neural network
    cnn2 = CNN2()

    # train the variant 2 architecture convolutional neural network
    cnn2_performance = cnn2.train(
        train_data_loader, learning_rate=0.01, n_epochs=20)

    # test the variant 2 architecture convolutional neural network
    cnn2.test(test_data_loader)

    # plot the test results
    plt.plot(range(20), mlp_performance, color='black', label='MLP')
    plt.plot(range(20), base_cnn_performance, color='red', label='Base CNN')
    plt.plot(range(20), cnn1_performance, color='green', label='CNN 1')
    plt.plot(range(20), cnn2_performance, color='blue', label='CNN 2')
    plt.title('Neural Network Image Classification Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
