import numpy as np
from sklearn import preprocessing

def estandar(dat):
    z_scaler = preprocessing.StandardScaler()
    return z_scaler.fit_transform(dat)

def minmax(dat):
    mm_scaler = preprocessing.MinMaxScaler()
    return mm_scaler.fit_transform(dat)

def maxabs(dat):
    ma_scaler = preprocessing.MaxAbsScaler
    return ma_scaler.fit_transform(dat)

def robust(dat):
    r_scaler = preprocessing.RobustScaler()
    return r_scaler.fit_transform(dat)

def DeNormalizacion(): # esto es algo asi pero le falta investigar
    return preprocessing.inverse_transform()

class Normalizacion():
    def __init__(self, normalizacion):
        self.normalizacion = normalizacion

    def Normalizacion(prediction, normalizacion):
        if normalizacion == "estandar":
            N = estandar(prediction)
        elif normalizacion == "minmax":
            N = minmax(prediction)
        elif normalizacion == "robust":
            N = maxabs(prediction)
        elif normalizacion == "norma":
            N = robust(prediction)
        return N
