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



def ex_2_2(input1, target1, input2, target2):
    """
    Solution for exercise 2.2
    :param input1: The input from dataset1
    :param target1: The target from dataset1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    train = input1
    test = input2
    target_train = target1[:, 1]
    target_test = target2[:, 1]


    ## TODO
    n_hidden_neurons = 20

    accu_list_train = np.zeros((10,1))
    accu_list_test = np.zeros((10, 1))

# Find the best seed
    for seed in range(10):
        nn = MLPClassifier(activation='tanh', solver='adam', max_iter=1000, hidden_layer_sizes=(n_hidden_neurons,),random_state=seed)
        nn.fit(train, target_train)
        accu_list_train[seed] = nn.score(train, target_train)
        accu_list_test[seed] = nn.score(test, target_test)

# Compute NN weights with the best seed
    best_seed = np.argmax(accu_list_train)
    best_nn = nn = MLPClassifier(activation='tanh', solver='adam', max_iter=1000, hidden_layer_sizes=(n_hidden_neurons,),random_state=best_seed)
    best_nn.fit(train, target_train)

# Evaluate the confusion matrix with best NN
    predictions = nn.predict(test)
    C = confusion_matrix(target_test, predictions)
    print(C)

# Plot results
    plot_histogram_of_acc(accu_list_train, accu_list_test)
    print(accu_list_test)
# Find misclassified images
#    comp_array = (target_test == predictions).all()
    comp_array = target_test - predictions
    print(comp_array)
    comp_vector2 = np.nonzero(comp_array)
    print(comp_vector2)

    plot_image(test(comp_vector2[1]))
    plot_image(test(comp_vector2[5]))
    plot_image(test(comp_vector2[8]))
    print(test(comp_vector2[1]))

# Plot misclassified image


    pass
