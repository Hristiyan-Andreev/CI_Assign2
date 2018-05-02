import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha, plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    # prediction = nn.predict(x)
    # E = 1/2*(prediction - y)**2
    # N = len(E)
    # mse = 1/N*sum(E)

    # why aren't we using the function mse_?
    mse = mean_squared_error(y, nn.predict(x))
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    for i in [2, 8, 40]:
        n_hidden_neurons = i
        nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=200, hidden_layer_sizes=(n_hidden_neurons,), alpha=0)
        nn.fit(x_train, y_train)
        predictions_test = nn.predict(x_test)
        plot_learned_function(n_hidden_neurons, x_train, y_train, 0, x_test, y_test, predictions_test)
        plt.show()

def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    mse_list_train = np.zeros(10,)
    mse_list_test = np.zeros(10,)
    for i in range(10):
        n_hidden_neurons = 8
        nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=200, hidden_layer_sizes=(n_hidden_neurons,),
                          alpha=0, random_state=i)
        nn.fit(x_train, y_train)
        mse_list_train[i] = calculate_mse(nn, x_train, y_train)
        mse_list_test[i] = calculate_mse(nn, x_test, y_test)

    mse_test_std = np.std(mse_list_test)
    mse_test_avg = np.average(mse_list_test)
    mse_train_std = np.std(mse_list_train)
    mse_train_avg = np.average(mse_list_train)

    print(mse_list_train,'Train avg:',mse_train_avg,'Train STD:', mse_train_std)
    print(mse_list_test, 'Test avg:',mse_test_avg,'Test STD:', mse_test_std)

def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    Use max_iter = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    hidden_neurons_totest = np.array([1, 2, 3, 4, 6, 8, 12, 20, 40])
    # hidden_neurons_totest = np.array([20])
    dim1 = hidden_neurons_totest.shape[0]
    mse_test_matrix = np.zeros((dim1, 10))
    mse_train_matrix = np.zeros((dim1, 10))
    k = 0
    for i in hidden_neurons_totest:
        n_hidden_neurons = i
        for j in range(10):
            nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=10000, tol=1e-8,
                              hidden_layer_sizes=(n_hidden_neurons,), alpha=0, random_state=j)
            nn.fit(x_train, y_train)
            predictions_test = nn.predict(x_test)
            mse_test_matrix[k, j] = calculate_mse(nn, x_test, y_test)
            mse_train_matrix[k, j] = calculate_mse(nn, x_train, y_train)
        k += 1
    plot_mse_vs_neurons(mse_train_matrix, mse_test_matrix, hidden_neurons_totest)
    plt.show()
    plot_learned_function(40, x_train, y_train, 0, x_test, y_test, predictions_test)
    plt.show()

def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 d)
    Remember to set alpha to 0 when initializing the model
    Use n_iterations = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    hidden_neurons_totest = np.array([2, 8, 40])
    hidden_neurons_list = [2, 8, 40]
    dim1 = hidden_neurons_totest.shape[0]
    mse_test_matrix = np.zeros((dim1, 10000))
    mse_train_matrix = np.zeros((dim1, 10000))
    k = 0
    for i in hidden_neurons_totest:
        n_hidden_neurons = i
        nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=1, tol=1e-8,
                          hidden_layer_sizes=(n_hidden_neurons,), alpha=0, random_state=0, warm_start=True)
        # test again with solver='adam' and 'sgd'
        for j in range(10000):
            nn.fit(x_train, y_train)
            mse_test_matrix[k, j] = calculate_mse(nn, x_test, y_test)
            mse_train_matrix[k, j] = calculate_mse(nn, x_train, y_train)
        k += 1
    plot_mse_vs_iterations(mse_train_matrix, mse_test_matrix, 10000, hidden_neurons_totest)
    plt.show()

def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    alpha_values = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    dim1 = alpha_values.shape[0]
    mse_test_matrix = np.zeros((dim1, 10))
    mse_train_matrix = np.zeros((dim1, 10))
    k = 0
    for i in alpha_values:
        alpha = i
        for j in range(10):
            nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=200,
                              hidden_layer_sizes=(40,), alpha=i, random_state=j)
            nn.fit(x_train, y_train)
            predictions_test = nn.predict(x_test)
            mse_test_matrix[k, j] = calculate_mse(nn, x_test, y_test)
            mse_train_matrix[k, j] = calculate_mse(nn, x_train, y_train)
        k += 1
    plot_mse_vs_alpha(mse_train_matrix, mse_test_matrix, alpha_values)
    plt.show()

def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    # validation set generation
    h = x_train.shape[0]
    train_indexes = range(h)
    index_perm = np.random.permutation(train_indexes)
    index_train_new = index_perm[0:int(np.floor(h/2))]
    index_validation = index_perm[int(np.floor(h/2)):]
    x_train_new = x_train[index_train_new]
    y_train_new = y_train[index_train_new]
    x_validation = x_train[index_validation]
    y_validation = y_train[index_validation]
    mse_test_matrix = np.zeros((10, 100))
    mse_train_matrix = np.zeros((10, 100))
    mse_validation_matrix = np.zeros((10, 100))
    #-------

    k = 0
    for i in range(10):
        nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=20,
                          hidden_layer_sizes=(40,), alpha=1e-3, random_state=i, warm_start=True)
        for j in range(100):
            nn.fit(x_train_new, y_train_new)
            mse_test_matrix[k, j] = calculate_mse(nn, x_test, y_test)
            mse_validation_matrix[k, j] = calculate_mse(nn, x_validation, y_validation)
            mse_train_matrix[k, j] = calculate_mse(nn, x_train_new, y_train_new)
        k = k+1

    test_mse_end = mse_test_matrix[:, 99]
    index_min = np.zeros((10, 1))
    test_mse_early_stopping = np.zeros((10, 1))
    test_mse_ideal = np.zeros((10, 1))
    for i in range(10):
        index_min[i] = np.argmin(mse_validation_matrix[i, :])
        test_mse_early_stopping[i] = mse_test_matrix[i, int(index_min[i])]
        test_mse_ideal[i] = np.min(mse_test_matrix[i, :])

    plot_bars_early_stopping_mse_comparison(test_mse_end, test_mse_early_stopping, test_mse_ideal)
    plt.show()

def ex_1_2_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    ## TODO
    ideal_hidden_neurons = 8  # all random, we must think about this
    ideal_alpha = 1e-2  #random
    ideal_solver = 'lbfgs'  #random

    # validation set generation
    h = x_train.shape[0]
    train_indexes = range(h)
    index_perm = np.random.permutation(train_indexes)
    index_train_new = index_perm[0:int(np.floor(h / 2))]
    index_validation = index_perm[int(np.floor(h / 2)):]
    x_train_new = x_train[index_train_new]
    y_train_new = y_train[index_train_new]
    x_validation = x_train[index_validation]
    y_validation = y_train[index_validation]
    mse_test_matrix = np.zeros((10, 100))
    mse_train_matrix = np.zeros((10, 100))
    mse_validation_matrix = np.zeros((10, 100))
    # -------

    k = 0
    j_index = np.zeros((10, 1))
    for i in range(10):
        nn = MLPRegressor(activation='logistic', solver=ideal_solver, max_iter=20,
                          hidden_layer_sizes=(ideal_hidden_neurons,), alpha=ideal_alpha, random_state=i, warm_start=True)
        for j in range(100):
            nn.fit(x_train_new, y_train_new)
            mse_test_matrix[k, j] = calculate_mse(nn, x_test, y_test)
            mse_validation_matrix[k, j] = calculate_mse(nn, x_validation, y_validation)
            mse_train_matrix[k, j] = calculate_mse(nn, x_train_new, y_train_new)
            if j > 1 and mse_validation_matrix[k, j] >= mse_validation_matrix[k, j-1]:
                j_index[i] = j-1
                mse_validation_matrix[k, j] = 0
                break
        k = k + 1

    mse_test_vector = np.array(())
    mse_validation_vector = np.array(())
    mse_train_vector = np.array(())
    for i in range(10):
        mse_test_vector = np.append(mse_test_vector, mse_test_matrix[i, 0:int(j_index[i])])
        mse_validation_vector = np.append(mse_validation_vector, mse_validation_matrix[i, 0:int(j_index[i])])
        mse_train_vector = np.append(mse_train_vector, mse_train_matrix[i, 0:int(j_index[i])])

    mse_test_mean = np.mean(mse_test_vector)
    mse_validation_mean = np.mean(mse_validation_vector)
    mse_train_mean = np.mean(mse_train_vector)
    mse_test_std = np.std(mse_test_vector)
    mse_validation_std = np.std(mse_validation_vector)
    mse_train_mean_std = np.std(mse_train_vector)

    optimal_choice = np.argmin(j_index)  # num. iteration = 20*j_index(optimal_choice)
    mse_test_mean_opt = np.mean(mse_test_matrix[optimal_choice, 0:int(j_index[optimal_choice])])
    mse_valid_mean_opt = np.mean(mse_validation_matrix[optimal_choice, 0:int(j_index[optimal_choice])])
    mse_train_mean_opt = np.mean(mse_train_matrix[optimal_choice, 0:int(j_index[optimal_choice])])
    mse_test_std_opt = np.std(mse_test_matrix[optimal_choice, 0:int(j_index[optimal_choice])])
    mse_valid_std_opt = np.std(mse_validation_matrix[optimal_choice, 0:int(j_index[optimal_choice])])
    mse_train_std_opt = np.std(mse_train_matrix[optimal_choice, 0:int(j_index[optimal_choice])])

    print(mse_test_mean, mse_validation_mean, mse_train_mean, mse_test_std,  mse_validation_std, mse_train_mean_std)
    print(mse_test_mean_opt, mse_valid_mean_opt, mse_train_mean_opt, mse_test_std_opt, mse_valid_std_opt, mse_train_std_opt)
    print(optimal_choice)
    print(j_index)

