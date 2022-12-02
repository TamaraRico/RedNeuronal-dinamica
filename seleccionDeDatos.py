import numpy as np
from sklearn import model_selection

def aleatorio(inputs, outputs, train_size, val_size, test_size):
    validation = val_size / (test_size + val_size)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(inputs, outputs, train_size=train_size)
    x_val, x_test, y_val, y_test = model_selection.train_test_split(x_test, y_test, train_size=validation)
    return x_train, y_train, x_val, y_val, x_test, y_test

def kfold(inputs, outputs):
    kf = model_selection.KFold(n_splits=5)
    for train_index, test_index in kf.split(inputs):
        x_train, x_test = inputs[train_index], inputs[test_index]
        y_train, y_test = outputs[train_index], outputs[test_index]
    for test_index, val_index in kf.split(x_test):
        x_test, x_val = inputs[test_index], inputs[val_index]
        y_test, y_val = outputs[test_index], outputs[val_index]
    return x_train, y_train, x_val, y_val, x_test, y_test

"""def biologico(inputs, outputs, train_size, val_size, test_size):
    validation = val_size / (test_size + val_size)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(inputs, outputs, train_size=train_size)
    x_val, x_test, y_val, y_test = model_selection.train_test_split(x_test, y_test, train_size=validation)
    return x_train, y_train, x_val, y_val, x_test, y_test"""


def seleccion(selection, inputs, outputs, train_size, val_size, test_size):
    if selection == "aleatorio":
        x_train, y_train, x_val, y_val, x_test, y_test = aleatorio(inputs, outputs, train_size, val_size, test_size)
    elif selection == "kfold":
        x_train, y_train, x_val, y_val, x_test, y_test = kfold(inputs, outputs)
    return x_train.T, y_train.T, x_val.T, y_val.T, x_test.T, y_test.T

