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

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # puede q esto no se ocupe
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def tanh_backward(dA, cache):
    Z = cache
    s,_ = tanh(Z)
    dZ = dA * (1-s**2)
    assert (dZ.shape == Z.shape)
    return dZ

def activation_layer(prediction, activation):
    if activation == "sigmoid":
        A = sigmoid(prediction)
    elif activation == "relu":
        A = relu(prediction)
    elif activation == "tanh":
        A = tanh(prediction)
    else:
        A = prediction
    # assert (A.shape == (W.shape[0], A_prev.shape[1]))
    return A

def derivative_activation_function(prediction, activation):
    if activation == "sigmoid":
        A = dsigmoid(prediction)
        # A = sigmoid_backward(prediction)
    elif activation == "relu":
        A = drelu(prediction)
        # A = sigmoid_backward(prediction)
    elif activation == "tanh":
        A = dtanh(prediction)
        # A = sigmoid_backward(prediction)
    return A