# CSE 5526: Lab 4 - Convolutional Neural Networks

This repository contains the codebase to construct and test three different PyTorch convolutional neural networks. Each convolutional neural network explored in this lab is tasked with performing image classification on the Fashion MNIST data set. After training each model and testing its performance on the test set, each model is then compared to one another to determine which convolutional neural network classifies the test set of the Fashion MNIST with the greatest accuracy. We then explore the differences of each model to compare and determine the effects of varying architectures on image classification performance.

## Getting Started

To get started with this repository, first clone this repository to your local machine and then install all the required python dependencies listed in the requirements.txt file. You can install the listed python dependencies with the following command.

    pip install -r requirements.txt

Next, to train and test each convolutional neural network, simply run the main method with the following command.

    python main.py

The performance metrics of each model will be printed to the console and visually displayed with matplotlib graphs.
