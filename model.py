# -*- coding: utf-8 -*-

import numpy as np
import math as m
from activation import *
from weightAndBias import *
from utils import *
import time

# network flatten to list
def matrixToVector(derivadas):
    iteraciones = len(derivadas)
    vector = []
    for i in range(iteraciones):
        a = derivadas[i]
        v = a.flatten()
        vector.append(v)
        if(i > 0):
            lista = np.concatenate((vector[i-1], vector[i]), axis=0)
    vector = np.vstack(lista)
    return vector

def vectorToMatrix(vector, net):
    l = 0
    matrix = []
    # print('vector', vector)
    for i in range(net.capas-1):
        array = []
        for j in range(net.layers[i+1].neuronas):
            row = []
            for k in range(net.layers[i].neuronas+1):
                row.append(vector[l][0])
                l = l+1
            array.append(row)
        array = np.vstack(array)
        print('iter', array)
        matrix.append(array)
        print('matrix', matrix)
    return matrix

def RMSprop(net, weigths, derivadas, vt, eps, alpha, beta):
    newWeights = []
    weigth_vector = matrixToVector(weigths)
    derivadas_vector = matrixToVector(derivadas)
    vt = beta * vt + (1 - beta) * derivadas_vector ** 2.0
    weigth_vector = weigth_vector - alpha / (np.sqrt(vt + eps)) * derivadas_vector
    wv = vectorToMatrix(weigth_vector, net)
    for i in range(net.capas-1):
        w = wv[i]
        newWeights.append({'weights': w})
    return vt, newWeights

# Multicapa
def backpropagation_classification(net, AL, e):
    eta = net.learning_rate
    newWeights = []
    derivadas = []
    weights = []
    for l in range(net.capas - 2, -1, -1):
        a_prev = AL[l + 1]
        ae = np.concatenate((AL[l], np.ones((1, np.size(AL[l], 1)))), axis=0)
        df_dnet = derivative_activation_function(a_prev, net.layers[l+1].activation_function)
        derivadas.append(df_dnet)
        if(l == net.capas - 2): # poner las deltas en un arreglo
            delta = df_dnet * (-2 * e)
        else:
            delta = df_dnet * (np.matmul((net.weightsbias[l + 1].get('weights')[:, 0:-1]).T, delta))
        W = net.weightsbias[l].get('weights') - eta * (np.matmul(delta, ae.T))
        newWeights.append({'weights': W})
        print(newWeights)
        weights.append(W)
    
    derivadas.reverse()
    newWeights.reverse()
    weights.reverse()
    return newWeights, derivadas, weights

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
    if(net.performanceFunction == 'cross_entropy'):
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
        df_dnet = derivative_activation_function(a_prev, net.layers[l+1].activation_function)
        if(l == net.capas - 2): # poner las derivadas en un arreglo
            if(net.performanceFunction == 'cross_entropy'):
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
        print('backprop',newWeights)
        weights.append(W)
    
    derivadas.reverse()
    newWeights.reverse()
    weights.reverse()
    return newWeights, derivadas, weights

# entropia cruzada => delta = error 

# Multicapas
def train(net, epochs):
    for epoch in range(epochs):
        a, e, p = forward(net) # plot p
        print('net', net.weightsbias)
        net.weightsbias, derivatives, w = backpropagation(net, a, e)
        if epoch == 0: # primera iteracion 
            vt, net.weightsbias = RMSprop(net, w, derivatives, 0, 1e-8, 0.001, 0.9)
        else:
            vt, net.weightsbias = RMSprop(net, w, derivatives, vt, 1e-8, 0.001, 0.9)
        print('W', net.weightsbias)
    # print('derivadas', grads)
    # print('matriz', w)
    # print('vector to matrix', matrixToVector(w))
    # print('matrix to vector', vectorToMatrix(matrixToVector(w), net), '\n')
    return a, e


# Dos capas
def train2(a0, T, W10, W21, epochs, lr):
    for epoch in range(epochs):
        a2, a1, e = forward2(W10, W21, a0, T);
        W10, W21 = backward(W10, W21, e, a0, a1, a2, lr);
    return a2, W10, W21, e


class net:
    def __init__(self, layers, activation_function, perFunc, inputs, outputs, lr):
        self.learning_rate = lr
        self.performanceFunction = perFunc
        #Numero de capas totales
        self.capas = np.size(layers, 0) + 2

        #Arreglo de capas
        self.layers = []
        rowsin, columns = np.shape(inputs)
        self.layers.append(Layer(rowsin, 'null')) #Agrega la capa de entrada con funcion de activacion null
        rows, columns = np.shape(outputs) #Se agrega el numero de neuronas de la ultima capa al arreglo par inicializar la capa
        layers.append(rows)
        for c in range(self.capas - 1):
            self.layers.append(Layer(layers[c], activation_function[c]))  #Agrega las capas ocultas y la de salida

        self.inputs = inputs
        self.target = outputs
        
        # Se llama a la funcion en weightAndBias para inicializarlos
        # self.initializeWB
        self.weightsbias = []

        for l in range(1, self.capas):
            if l == 1:
                weigths, bias = initnguyenwidrow(rowsin, layers[l-1])
            else:
                weigths, bias = initnguyenwidrow(layers[l - 2], layers[l - 1])
            W = np.concatenate((weigths, bias), axis=1)
            wb = {'weights': W}
            self.weightsbias.append(wb)

        # Se llama a la funcion en seleccionDeDatos para separarlos
        """ self.divideFcn
        self.train
        self.val
        self.test

        # Funcion de desempeño
        self.performFcn
        """

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