from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc
import numpy as np

__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):
    """
    Solution for exercise 2.1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    ## TODO
    n_hidden_neurons = 6
    nn = MLPClassifier(activation='tanh', solver='adam', max_iter=200, hidden_layer_sizes=(n_hidden_neurons,))
    target = target2[:,2]
    ## Train the network
    nn.fit(input2, target)
    predictions = nn.predict(input2)
    C=confusion_matrix(target,predictions)
    hidden_layer_weights = nn.coefs_[0]
    plot_hidden_layer_weights(hidden_layer_weights)
    print(C)
    pass


def ex_2_2(input1, target1, input2, target2):
    """
    Solution for exercise 2.2
    :param input1: The input from dataset1
    :param target1: The target from dataset1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    ## TODO
    pass
