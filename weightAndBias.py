import numpy as np


def initnguyenwidrow(N0, N1):
    beta = 0.7 * N1 ** (1 / N0);
    W = -0.5 + np.random.randint(0, high=1, size=(N1, N0));
    for i in range(N1):
        W[i,:] = beta * W[i,:] / np.linalg.norm(W[i,:]);
    b = -beta + 2 * beta * np.random.randint(0, high=1, size=(N1, 1));
    return W, b
