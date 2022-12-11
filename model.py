# -*- coding: utf-8 -*-

import numpy as np
import math as m
from activation import *
from weightAndBias import *
from utils import *
import time
from seleccionDeDatos import *


# network flatten to list
def matrixToVector(derivadas):
    iteraciones = len(derivadas)
    vector = []
    for i in range(iteraciones):
        a = derivadas[i]
        v = a.flatten()
        vector.append(v)
        if (i == 1):
            lista = np.concatenate((vector[i - 1], vector[i]), axis=0)
        if (i > 1):
            lista = np.concatenate((lista, vector[i]), axis=0)
    vector = np.vstack(lista)
    return vector


def vectorToMatrix(vector, net):
    l = 0
    matrix = []
    # print('vector', vector)
    for i in range(net.capas - 1):
        array = []
        for j in range(net.layers[i + 1].neuronas):
            row = []
            for k in range(net.layers[i].neuronas + 1):
                row.append(vector[l][0])
                l = l + 1
            array.append(row)
        array = np.vstack(array)
        #print('iter', array)
        matrix.append(array)
        #print('matrix', matrix)
    return matrix


def RMSprop(net, weigths, derivadas, vt, eps, alpha, beta):
    newWeights = []
    weigth_vector = matrixToVector(weigths)
    derivadas_vector = matrixToVector(derivadas)
    vt = beta * vt + (1 - beta) * derivadas_vector ** 2.0
    weigth_vector = weigth_vector - alpha * (derivadas_vector / (np.sqrt(vt)+ eps))
    wv = vectorToMatrix(weigth_vector, net)
    for i in range(net.capas - 1):
        w = wv[i]
        newWeights.append({'weights': w})
    return vt, newWeights


def backpropagation_classification2(net, AL, e):
    eta = net.learning_rate
    newWeights = []
    derivadas = []
    weights = []
    for l in range(net.capas - 2, -1, -1):
        a_prev = AL[l + 1]
        ae = np.concatenate((AL[l], np.ones((1, np.size(AL[l], 1)))), axis=0)
        df_dnet = derivative_activation_function(a_prev, net.layers[l + 1].activation_function)
        if (l == net.capas - 2):  # poner las deltas en un arreglo
            delta = e
        else:
            delta = df_dnet * (np.matmul((net.weightsbias[l + 1].get('weights')[:, 0:-1]).T, delta))
        d = np.matmul(delta, ae.T)
        derivadas.append(d)
        W = net.weightsbias[l].get('weights') - eta * (np.matmul(delta, ae.T))
        newWeights.append({'weights': W})
        weights.append(W)

    derivadas.reverse()
    newWeights.reverse()
    weights.reverse()
    return newWeights, derivadas, weights

# Multicapa
def backpropagation_classification(net, AL, n, target):
    eta = net.learning_rate
    newWeights = []
    derivadas = []
    weights = []
    for l in range(net.capas - 2, -1, -1):
        if (l == net.capas - 2):  # poner las deltas en un arreglo
            a_prev = AL[l+1]
            delta = a_prev - target
        else:
            df_dnet = derivative_activation_function(n[l], net.layers[l + 1].activation_function)
            delta = df_dnet * (np.matmul((net.weightsbias[l + 1].get('weights')[:, 0:-1]).T, delta))
        ae = np.concatenate((AL[l], np.ones((1, np.size(AL[l], 1)))), axis=0)
        W = net.weightsbias[l].get('weights') - eta * (np.matmul(delta, ae.T))
        we = {'weights': W}
        newWeights.append(we)
        weights.append(W)
        df_weights = derivative_activation_function(W, net.layers[l + 1].activation_function)
        derivadas.append(df_weights)

    derivadas.reverse()
    newWeights.reverse()
    weights.reverse()
    return newWeights, derivadas, weights

# Multicapa
def forward_classification(net, inputs, target):
    a = []
    nets = []
    activation = inputs
    a.append(inputs)
    for l in range(1, net.capas):
        activatione = np.concatenate((activation, np.ones((1, np.size(activation, 1)))), axis=0)
        n = np.matmul(net.weightsbias[l - 1].get('weights'), activatione)
        nets.append(n)
        if (l == (net.capas-1)):
            activation = activation_layer(n, net.layers[l].activation_function)
            activation = activation_layer(activation.T, 'softmax').T
        else:
            activation = activation_layer(n, net.layers[l].activation_function)
        a.append(activation)
    p = cost_function(target, activation, 'cross_entropy')
    e = activation - target
    # aqui se agrega la funcion de performance, mse, sse
    return a, nets, p, e

def train_classification(net, epochs):
    for epoch in range(epochs):
        print("Epoca: ", epoch)
        a, n, p, e = forward_classification(net, net.x_train, net.y_train)  # plot p
        net.weightsbias, derivatives, w = backpropagation_classification2(net, a, e)
        if epoch == 0:  # primera iteracion
            vt, net.weightsbias = RMSprop(net, w, derivatives, 0, 1e-8, 0.001, 0.9)
        else:
            vt, net.weightsbias = RMSprop(net, w, derivatives, vt, 1e-8, 0.001, 0.9)
    return a[net.capas-1]

# Multicapa
def forward(net):
    a = []
    a.append(net.inputs)
    activatione = np.concatenate((net.inputs, np.ones((1, np.size(net.inputs, 1)))), axis=0)
    for l in range(1, net.capas):
        n = np.matmul(net.weightsbias[l - 1].get('weights'), activatione)
        activation = activation_layer(n, net.layers[l].activation_function)
        activatione = np.concatenate((activation, np.ones((1, np.size(activation, 1)))), axis=0)
        a.append(activation)
    if (net.performanceFunction == 'cross_entropy'):
        e = activation - net.target
    else:
        e = net.target - activation
    p = cost_function(net.target, activation, net.performanceFunction)
    # aqui se agrega la funcion de performance, mse, sse
    return a, e, p


# Multicapa
def backpropagation(net, AL, e):
    eta = net.learning_rate
    newWeights = []
    derivadas = []
    weights = []
    for l in range(net.capas - 2, -1, -1):
        a_prev = AL[l + 1]
        ae = np.concatenate((AL[l], np.ones((1, np.size(AL[l], 1)))), axis=0)
        df_dnet = derivative_activation_function(a_prev, net.layers[l + 1].activation_function)
        if (l == net.capas - 2):  # poner las derivadas en un arreglo
            if (net.performanceFunction == 'cross_entropy'):
                delta = e
            else:
                delta = df_dnet * (-2 * e)
        else:
            delta = df_dnet * (np.matmul((net.weightsbias[l + 1].get('weights')[:, 0:-1]).T, delta))
        d = np.matmul(delta, ae.T)
        derivadas.append(d)
        W = net.weightsbias[l].get('weights') - eta * d
        # derivada (np.matmul(delta, ae.T)) = este deberia ser pasado al rms prop con los pesos
        newWeights.append({'weights': W})
        #print('backprop', newWeights)
        weights.append(W)

    derivadas.reverse()
    newWeights.reverse()
    weights.reverse()
    return newWeights, derivadas, weights


# entropia cruzada => delta = error

# Multicapas
def train(net, epochs):
    for epoch in range(epochs):
        a, e, p = forward(net)  # plot p
        net.weightsbias, derivatives, w = backpropagation(net, a, e)
        if epoch == 0:  # primera iteracion
            vt, net.weightsbias = RMSprop(net, w, derivatives, 0, 1e-8, 0.001, 0.9)
        else:
            vt, net.weightsbias = RMSprop(net, w, derivatives, vt, 1e-8, 0.001, 0.9)

        print('net2', net.weightsbias)
    return a


class net:
    def __init__(self, layers, activation_function, perFunc, divideFunc, inputs, outputs, lr):
        self.learning_rate = lr
        self.performanceFunction = perFunc
        # Numero de capas totales
        self.capas = np.size(layers, 0) + 2

        self.inputs = inputs
        self.target = outputs

        # Se llama a la funcion en seleccionDeDatos para separarlos
        self.divideFcn = divideFunc
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = seleccion(divideFunc,
                                                                                                 self.inputs,
                                                                                                 self.target, 0.50,
                                                                                                 0.25, 0.25)

        # Arreglo de capas
        self.layers = []
        rowsin, columns = np.shape(self.x_train)
        self.layers.append(Layer(rowsin, 'null'))  # Agrega la capa de entrada con funcion de activacion null
        rows, columns = np.shape(
            self.y_train)  # Se agrega el numero de neuronas de la ultima capa al arreglo par inicializar la capa
        layers.append(rows)
        for c in range(self.capas - 1):
            self.layers.append(Layer(layers[c], activation_function[c]))  # Agrega las capas ocultas y la de salida

        # Se llama a la funcion en weightAndBias para inicializarlos
        # self.initializeWB
        self.weightsbias = []

        for l in range(1, self.capas):
            if l == 1:
                weigths, bias = initnguyenwidrow(rowsin, layers[l - 1])
            else:
                weigths, bias = initnguyenwidrow(layers[l - 2], layers[l - 1])
            W = np.concatenate((weigths, bias), axis=1)
            wb = {'weights': W}
            self.weightsbias.append(wb)

        # Epocas maximas
        self.epochs = 500


class Layer:
    def __init__(self, neuronas, activation):
        self.neuronas = neuronas
        self.activation_function = activation

    def activationFun(self):
        return self.activation_function

    def numNeurons(self):
        return self.neuronas