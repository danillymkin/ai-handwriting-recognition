import numpy as np


def soft_max(t):
    # get exp vector and normalization
    out = np.exp(t)
    return out / np.sum(out)


def relu(t):
    return np.maximum(t, 0)


def relu_deriv(t):
    return (t >= 0).astype(float)


def sigmoid(t, a):
    return 1 / (1 + np.exp(-t * a))


def sigmoid_deriv(t, a):
    return a * (np.exp(-t * a) / ((1 + np.exp(-t * a)) ** 2))


def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def to_full(y, classes_count):
    y_full = np.zeros((1, classes_count))
    y_full[0, y] = 1
    return y_full
