# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # 1 vs 1.0

def dsigmoid(x):
    return 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dtanh(x):
    return 1 - ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))) ** 2

def softmax(x):
    assert len(x.shape) == 2

    s = np.max(x, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    res = e_x / div
    return e_x / div

def activation_layer(prediction, activation):
    if activation == "sigmoid":
        A = sigmoid(prediction)
    elif activation == "relu":
        A = relu(prediction)
    elif activation == "tanh":
        A = tanh(prediction)
    elif activation == "softmax":
        A = softmax(prediction)
    else:
        A = prediction
    return A

def derivative_activation_function(prediction, activation):
    if activation == "sigmoid":
        A = dsigmoid(prediction)
    elif activation == "relu":
        A = drelu(prediction)
    elif activation == "tanh":
        A = dtanh(prediction)
    elif activation == "softmax":
        A = 0
    return A