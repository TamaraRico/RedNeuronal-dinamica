# -*- coding: utf-8 -*-

from pretty_confusion_matrix import pp_matrix_from_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import roc_curve
import math

def plot_linealRegression(Targets, Predicted):
    r = np.corrcoef(Targets.T[0], Predicted.T[0])
    plt.plot(Targets.T[0], Predicted.T[0], 'o')
    m, b = np.polyfit(Targets.T[0], Predicted.T[0], 1)
    plt.plot(Targets.T[0], m*Targets.T[0] + b)
    plt.title("R={}".format(r[0][1]))
    str = "{}*{}={}".format(round(m, 3), "Target",round(b, 3))
    plt.ylabel(str)
    plt.show()

def getClasses_Classification(Targets):
    if Targets.shape[1] == 1:
        obtainedClasses = np.zeros((Targets.shape[0],1))
        classes_unique = np.unique(Targets)
        for class_value in classes_unique:
            obtainedClasses = np.hstack((obtainedClasses,
                np.array(
                    Targets == class_value, dtype=np.int32
                    )
            ))
        return obtainedClasses[:,1:]
    else:
        return Targets

def plot_logisticRegression_Classification(Targets, Ph):
    pp_matrix_from_data(np.argmax(Targets, axis = 1), np.argmax(Ph, axis = 1))
    plt.figure()
    color = iter(cm.rainbow(np.linspace(0,1,15)))

    for class_v in range(Targets.shape[1]):
        fpr, tpr, none = roc_curve(Targets[:,class_v].reshape(-1,1), Ph[:,class_v].reshape(-1,1))
        plt.plot(fpr, tpr, color=next(color), lw=1, label=f'Class {class_v}')

    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def cross_entropy(y,yh):
  loss =- np.sum(y * np.log(yh))
  return loss/float(yh.shape[0])

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
    elif function == 'cross_entropy':
        cost = cross_entropy(prediction, expected_output)
    return cost