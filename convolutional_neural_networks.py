"""
Convolutional Neural Networks Module
"""

# import necessary modules
import torch.nn as nn


class BaseCNN(nn.Module):
    """
    Base Architecture Convolutional Neural Network

    This class contains the code to construct the base architecture
    convolutional neural network as described in the lab write up. This
    convolutional neural network consists of two convolutional layers and
    three fully-connected linear layers with a softmax ouput.
    """

    def __init__(self):

        # base class constructor
        super(BaseCNN, self).__init__()

        # layer 1 - convolution, relu, max pooling
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # layer 2 - convolution, relu, max pooling
        self.layer_2 = nn.Sequential(
            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # layer 3 - fully connected linear layer with relu activation
        self.layer_3 = nn.Sequential(
            nn.Linear(4, 120),
            nn.ReLU()
        )

        # layer 4 - fully connected linear layer with relu activation
        self.layer_4 = nn.Sequential(
            nn.Linear(120, 60),
            nn.ReLU()
        )

        # layer 5 - fully connected linear layer with softmax activation
        self.layer_5 = nn.Sequential(
            nn.Linear(60, 10),
            nn.Softmax()
        )

    def forward(self, x):
        """
        Forward
        """

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        return x


class CNN1(nn.Module):
    """
    Variant 1 Architecture Convolutional Neural Network
    """

    def __init__(self):

        # base class constructor
        super(CNN1, self).__init__()

    def forward(self, x):

        return x


class CNN2(nn.Module):
    """
    Variant 2 Architecture Convolutional Neural Network
    """

    def __init__(self):

        # base class constructor
        super(CNN2, self).__init__()

    def forward(self, x):

        return x
