# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as plt
from normalizacion import *
from weightAndBias import *
from model import *

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

# Dos capas
# W10,b1 = initnguyenwidrow(N0,N1);
# W10E = np.concatenate((W10, b1), axis=1)
# W21,b2 = initnguyenwidrow(N1,N2);
# W21E = np.concatenate((W21, b2), axis=1)
# outn, W10E, W21E, e = train2(xn,tn,W10E,W21E,epochs,lr)
# print(outn)
# print(e)

capas = [{'capas': N0, 'activacion': 'relu'}, 
         {'capas': N1, 'activacion': 'relu'}, 
         {'capas': N2, 'activacion': 'sigmoid'}]

#print(capas)

#Multicapa
net = net([N1], ['relu', 'sigmoid'], xn, tn, lr=0.01)
# net = net(capas, xn, tn, lr=0.01)
a, e = train(net, epochs)
print(a)
print(e)

# agregar las derivadas
# hacer una clase de las funciones
# una funcion para elegir una tecnica para asignar los pesos
# las funciones de normalizacion
# como tomar los datos (lotes, mini lotes, estocastico)
# back propagation
# tener la opcion sin normalizar o normalizar

#Cuando se realiza la validacion por lo regular se admiten 6 fallas maximo (que sea peor), despues de eso para
#Cuando el gradiente se acerca mucho a 0 o incrementa demasiado se tiene que hacer una parada temprana
