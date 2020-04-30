"""
Multilayer Perceptron Module
"""

# import necessary modules
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import utils


class MLP(nn.Module):

    def __init__(self):

        # base class constructor
        super(MLP, self).__init__()

        # define two fully connected linear layers
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=10)

    def forward(self, feature):
        """
        Forward

        This method feeds the given feature forward through the neural network
        to determine the predicted target class.

        Args:
            feature (torch.Tensor): the given feature

        Returns:
            prediction (torch.Tensor): the predicted target class
        """

        # reshape the given feature
        feature = feature.reshape(-1, 28 * 28)

        # feed forward the feature
        prediction = self.fc1(feature)
        prediction = func.relu(prediction)
        prediction = self.fc2(prediction)

        return prediction

    def train(self, data_loader, learning_rate=0.01, n_epochs=1):
        """
        Train

        This method trains the neural network on the given features and labels
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
                labels = batch[1]

                predictions = self(images)

                loss = func.cross_entropy(predictions, labels)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            epoch_accuracy = utils.get_accuracy(self, data_loader) * 100
            print(f'epoch {epoch}: train set accuracy {epoch_accuracy:.3f}%')
