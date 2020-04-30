"""
Convolutional Neural Networks Module
"""

# import necessary modules
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import utils


class BaseCNN(nn.Module):
    """
    Base Architecture Convolutional Neural Network

    This class contains the code to construct the base architecture
    convolutional neural network as described in the lab write up. This
    convolutional neural network consists of two convolutional layers and
    three fully connected linear layers.
    """

    def __init__(self):

        # base class constructor
        super(BaseCNN, self).__init__()

        # layer 1 - convolution, relu, max pooling
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # layer 2 - convolution, relu, max pooling
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # layer 3 - fully connected linear layer with relu activation
        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=12 * 4 * 4, out_features=120),
            nn.ReLU()
        )

        # layer 4 - fully connected linear layer with relu activation
        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=120, out_features=60),
            nn.ReLU()
        )

        # layer 5 - fully connected linear layer
        self.layer_5 = nn.Sequential(
            nn.Linear(in_features=60, out_features=10)
        )

    def forward(self, feature):
        """
        Forward

        This method feeds the given feature forward through the neural network
        to determine a predicted target class given the feature.

        Args:
            feature (torch.Tensor): the feature

        Returns:
            prediction (torch.Tensor): the predicted target class
        """

        # reshape the feature
        feature = feature.reshape(-1, 1, 28, 28)

        # feed forward the feature
        prediction = self.layer_1(feature)
        prediction = self.layer_2(prediction)

        # reshape the prediction for the fully connected layers
        prediction = prediction.reshape(-1, 12 * 4 * 4)

        # feed forward the feature through fully connected layers
        prediction = self.layer_3(prediction)
        prediction = self.layer_4(prediction)
        prediction = self.layer_5(prediction)

        return prediction

    def train(self, data_loader, learning_rate=0.01, n_epochs=1):
        """
        Train

        This method trains the neural network on the given features and targets
        provided by the data loader batch processor.

        Args:
            data_loader (torch.utils.data.DataLoader): the data loader
            learning_rate (float): the learning rate hyperparameter
            n_epochs (int): the desired number of epochs to train for
        """

        # instantiate optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # train the model for a given number of epochs
        for epoch in range(n_epochs):
            for batch in data_loader:
                images = batch[0]
                targets = batch[1]

                # calculate predicted targets
                predictions = self(images)

                # calculate the loss
                loss = func.cross_entropy(predictions, targets)

                # clear the gradients
                optimizer.zero_grad()

                # backpropagate the loss
                loss.backward()

                # update weights and biases
                optimizer.step()

            # calculate the train set accuracy after each epoch
            epoch_accuracy = utils.get_accuracy(self, data_loader) * 100
            print(f'epoch {epoch}: train set accuracy {epoch_accuracy:.3f}%')

    def test(self, data_loader):
        """
        Test

        This method tests the neural network and reports the accuracy of
        the feature classification on the test set.

        Args:
            data_loader (torch.utils.data.DataLoader): the data loader

        Returns:
            accuracy (float): the accuracy of the feature classifications
        """

        # calculate test set accuracy
        accuracy = utils.get_accuracy(self, data_loader) * 100
        print(f'test set accuracy {accuracy:.3f}%')

        return accuracy


class CNN1(nn.Module):
    """
    Variant 1 Architecture Convolutional Neural Network

    This class contains the code to initialize, train, and test a convolutional
    neural network on the classification of the Fashion MNIST data set. The
    architecture of this convolutional neural network varies from the base
    architecture convolutional neural network class to explore the effects of
    architecture design on image classification accuracy.
    """

    def __init__(self):

        # base class constructor
        super(CNN1, self).__init__()

        # layer 1 - convolution, relu, max pooling
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        # layer 2 - convolution, relu, max pooling
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24,
                      kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        # layer 3 - fully connected linear layer with relu activation
        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=24 * 4 * 4, out_features=150),
            nn.ReLU()
        )

        # layer 4 - fully connected linear layer with relu activation
        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=150, out_features=50),
            nn.ReLU()
        )

        # layer 5 - fully connected linear layer
        self.layer_5 = nn.Sequential(
            nn.Linear(in_features=50, out_features=10)
        )

    def forward(self, feature):
        """
        Forward

        This method feeds the given feature forward through the neural network
        to determine a predicted target class given the feature.

        Args:
            feature (torch.Tensor): the feature

        Returns:
            prediction (torch.Tensor): the predicted target class
        """

        # reshape the feature
        feature = feature.reshape(-1, 1, 28, 28)

        # feed forward the feature
        prediction = self.layer_1(feature)
        prediction = self.layer_2(prediction)

        # reshape the prediction for the fully connected layers
        prediction = prediction.reshape(-1, 24 * 4 * 4)

        # feed forward the feature through fully connected layers
        prediction = self.layer_3(prediction)
        prediction = self.layer_4(prediction)
        prediction = self.layer_5(prediction)

        return prediction

    def train(self, data_loader, learning_rate=0.01, n_epochs=1):
        """
        Train

        This method trains the neural network on the given features and targets
        provided by the data loader batch processor.

        Args:
            data_loader (torch.utils.data.DataLoader): the data loader
            learning_rate (float): the learning rate hyperparameter
            n_epochs (int): the desired number of epochs to train for
        """

        # instantiate optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # train the model for a given number of epochs
        for epoch in range(n_epochs):
            for batch in data_loader:
                images = batch[0]
                targets = batch[1]

                # calculate predicted targets
                predictions = self(images)

                # calculate the loss
                loss = func.cross_entropy(predictions, targets)

                # clear the gradients
                optimizer.zero_grad()

                # backpropagate the loss
                loss.backward()

                # update weights and biases
                optimizer.step()

            # calculate the train set accuracy after each epoch
            epoch_accuracy = utils.get_accuracy(self, data_loader) * 100
            print(f'epoch {epoch}: train set accuracy {epoch_accuracy:.3f}%')

    def test(self, data_loader):
        """
        Test

        This method tests the neural network and reports the accuracy of
        the feature classification on the test set.

        Args:
            data_loader (torch.utils.data.DataLoader): the data loader

        Returns:
            accuracy (float): the accuracy of the feature classifications
        """

        # calculate test set accuracy
        accuracy = utils.get_accuracy(self, data_loader) * 100
        print(f'test set accuracy {accuracy:.3f}%')

        return accuracy


class CNN2(nn.Module):
    """
    Variant 2 Architecture Convolutional Neural Network

    This class contains the code to initialize, train, and test a convolutional
    neural network to classify images of clothing contained in the Fashion
    MNIST data set. The architecture of this convolutional neural network
    differs from the base architecture so that the effects of varying
    architectures on convolutional neural network image classifiers can be
    explored.
    """

    def __init__(self):

        # base class constructor
        super(CNN2, self).__init__()

        # layer 1 - convolution, relu, max pooling
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # layer 2 - fully connected linear layer with relu activation
        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=12 * 12 * 12, out_features=200),
            nn.ReLU()
        )

        # layer 3 - fully connected linear layer with relu activation
        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU()
        )

        # layer 4 - fully connected linear layer
        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=100, out_features=10)
        )

    def forward(self, feature):
        """
        Forward

        This method feeds the given feature forward through the neural network
        to determine a predicted target class given the feature.

        Args:
            feature (torch.Tensor): the feature

        Returns:
            prediction (torch.Tensor): the predicted target class
        """

        # reshape the feature
        feature = feature.reshape(-1, 1, 28, 28)

        # feed forward the feature
        prediction = self.layer_1(feature)

        # reshape the prediction for the fully connected layers
        prediction = prediction.reshape(-1, 12 * 12 * 12)

        # feed forward the feature through fully connected layers
        prediction = self.layer_2(prediction)
        prediction = self.layer_3(prediction)
        prediction = self.layer_4(prediction)

        return prediction

    def train(self, data_loader, learning_rate=0.01, n_epochs=1):
        """
        Train

        This method trains the neural network on the given features and targets
        provided by the data loader batch processor.

        Args:
            data_loader (torch.utils.data.DataLoader): the data loader
            learning_rate (float): the learning rate hyperparameter
            n_epochs (int): the desired number of epochs to train for
        """

        # instantiate optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # train the model for a given number of epochs
        for epoch in range(n_epochs):
            for batch in data_loader:
                images = batch[0]
                targets = batch[1]

                # calculate predicted targets
                predictions = self(images)

                # calculate the loss
                loss = func.cross_entropy(predictions, targets)

                # clear the gradients
                optimizer.zero_grad()

                # backpropagate the loss
                loss.backward()

                # update weights and biases
                optimizer.step()

            # calculate the train set accuracy after each epoch
            epoch_accuracy = utils.get_accuracy(self, data_loader) * 100
            print(f'epoch {epoch}: train set accuracy {epoch_accuracy:.3f}%')

    def test(self, data_loader):
        """
        Test

        This method tests the neural network and reports the accuracy of
        the feature classification on the test set.

        Args:
            data_loader (torch.utils.data.DataLoader): the data loader

        Returns:
            accuracy (float): the accuracy of the feature classifications
        """

        # calculate test set accuracy
        accuracy = utils.get_accuracy(self, data_loader) * 100
        print(f'test set accuracy {accuracy:.3f}%')

        return accuracy
