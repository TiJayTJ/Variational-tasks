import numpy as np


def phi(x):
    return np.sin(np.pi * x * (x - 1) / 3)


def psi(x, p):
    return np.sqrt(2) * np.sin(np.pi * x * p)


def sigma(h_tau, h_tau_index):
    return np.array([0, 1, 0.5, 0.5 - h_tau[h_tau_index][0] ** 2 / (12 * h_tau[h_tau_index][1])])