# -*- coding: utf-8 -*-

import numpy as np
import math


def initialize_parameters(layers):
    # parameters = {}
    w = []
    b = []
    no_layers = len(layers)
    for n in range(1, no_layers):
        w.append(np.random.randn(layers[n].neurons, layers[n - 1].neurons) * 0.01)
        b.append(np.zeros(shape=(layers[n].neurons, 1)))

        assert (w[n - 1].shape == (layers[n].neurons, layers[n - 1].neurons))
        assert (b[n - 1].shape == (layers[n].neurons, 1))

        # parameters['W' + str(n)] = np.random.randn(layers[n], layers[n-1]) * 0.01
        # parameters['b' + str(n)] = np.zeros(shape=(layers[n], 1))

        # assert(parameters['W' + str(n)].shape == (layers[n], layers[n-1]))
        # assert(parameters['b' + str(n)].shape == (layers[n], 1))

    return w, b


def RMSprop():
    return


def RMSprop_Optimization(X, Y, Alpha, Beta, num_iter, goal):
    n, q = X.shape
    size_theta = int(1 / 2 * (q + 1) * (q + 2))
    theta = np.zeros((size_theta, n))
    [A, thetaHat] = RMSprop(X, Y, theta, Alpha, Beta, num_iter, goal)
    yh = A @ thetaHat
    e = Y - yh
    RMSE = rmse(Y, yh)

    return thetaHat, RMSE, yh, e, A


def tanh(Z):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # 1 vs 1.0


def relu(x):
    return np.maximum(0, x)


def linear_forward(a, w, b):
    n = np.dot(w, a) + b
    assert (n.shape == (w.shape[0], a.shape[1]))
    cache = (a, w, b)
    return n, cache


def activation_layer(prediction, activation):
    if activation == "sigmoid":
        A = sigmoid(prediction)
    elif activation == "relu":
        A = relu(prediction)
    # assert (A.shape == (W.shape[0], A_prev.shape[1]))
    return A


def cross_entropy(y, yh):
    m = yh.shape[1]
    return (-1 / m) * np.sum(
        np.multiply(yh, np.log(y)) + np.multiply((1 - yh), np.log(1 - y)))  # creo que y y yh estan al reves


def sse(y, yh):
    return np.sum((y - yh) ** 2)


def mse(y, yh):
    return np.square(np.subtract(y, yh)).mean()


def rmse(y, yh):
    MSE = mse(y, yh)
    RMSE = math.sqrt(MSE)
    return RMSE


def cost_function(prediction, expected_output, function):
    if function == 'sse':
        cost = sse(prediction, expected_output)
    elif function == 'mse':
        cost = mse(prediction, expected_output)
    elif function == 'rmse':
        cost = rmse(prediction, expected_output)
    # cost = np.squeeze(cost)
    return cost


def compute_cost(AL, Y):  # cross entropy loss
    m = Y.shape[1]
    cost = (1. / m) * (-np.dot(Y, np.log(AL + pow(10.0, -9)).T) - np.dot(1 - Y, np.log(1 - AL + pow(10.0, -9)).T))
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())
    return cost
