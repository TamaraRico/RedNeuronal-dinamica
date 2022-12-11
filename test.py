# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as plt
import pandas as pd
import utils
from normalizacion import *
from weightAndBias import *
from model import *
from utils import *

N0 = 2
N1 = 3
N2 = 2

input_vectors = np.array([[4.7, 6.0],
                          [6.1, 3.9],
                          [2.9, 4.2],
                          [7.0, 5.5]]).T

targets = np.array([[3.52, 4.02],
                    [5.43, 6.23],
                    [4.95, 5.76],
                    [4.70, 4.28]]).T

xn = minmax(input_vectors)
tn = minmax(targets)

epochs = 100
lr = 0.01
batchsize = 1

# file = pd.read_csv("price_dataset.dat", sep=" ", header=None)
# inputs = np.array(file.iloc[:, :-1])
# targets = np.array(file.iloc[:,-1:])

# xn = minmax(inputs)
# tn = minmax(targets)

# net = net([10, 10, 10], ['sigmoid', 'sigmoid', 'sigmoid', 'tanh'], 'mse', 'aleatorio', xn, tn, lr=0.001)
# a = train(net, epochs)
# print('targets: ', np.shape(net.y_train))
# print('outputs: ', np.shape(a))
# utils.plot_linealRegression(a.T, net.y_train.T)

file = pd.read_csv("dermatology.dat", sep=" ", header=None)
inputs = np.array(file.iloc[:, :-1])
Y = np.array(file.iloc[:,-1:])
targets = getClasses_Classification(Y)

xn = minmax(inputs)
tn = minmax(targets)

net = net([10, 10], ['sigmoid', 'tanh', 'sigmoid'], 'mse', 'aleatorio', xn, tn, lr=0.01)
a = train_classification(net, epochs)
print('targets: ', net.y_train)
print('outputs: ', a)

utils.plot_linealRegression(net.y_train.T, a.T)
utils.plot_logisticRegression_Classification(net.y_train.T, a.T)