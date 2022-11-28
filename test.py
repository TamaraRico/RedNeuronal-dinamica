# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as plt
import pandas as pd
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

xn = minmax(input_vectors);
tn = minmax(targets);

epochs = 500;
lr = 0.01;
batchsize = 1;

capas = [{'capas': N0, 'activacion': 'relu'}, 
         {'capas': N1, 'activacion': 'relu'}, 
         {'capas': N2, 'activacion': 'sigmoid'}]
#print(capas)

#Multicapa
net = net([ N1], ['relu', 'sigmoid'], 'mse', xn, tn, lr=0.01)
# net = net(capas, xn, tn, lr=0.01)
a, e = train(net, epochs)
print(a)
print(e)


file = pd.read_csv("dermatology.dat", sep=" ", header=None)
inputs = np.array(file.iloc[:, :-1])
Y = np.array(file.iloc[:,-1:])
targets = getClasses_Classification(Y)

# plots 

# agregar las derivadas -> DONE
# hacer una clase de las funciones
# una funcion para elegir una tecnica para asignar los pesos
# las funciones de normalizacion
# como tomar los datos (lotes, mini lotes, estocastico)
# back propagation -> DONE
# tener la opcion sin normalizar o normalizar

#Cuando se realiza la validacion por lo regular se admiten 6 fallas maximo (que sea peor), despues de eso para
#Cuando el gradiente se acerca mucho a 0 o incrementa demasiado se tiene que hacer una parada temprana
