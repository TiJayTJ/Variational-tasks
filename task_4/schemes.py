import numpy as np

from task_4.math_functions import *


def explicit_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)

    for i in range(1, k + 1):
        for j in range(1, n + 1):
            sol[i, j] = sol[i - 1, j] + tau * (-c * (sol[i - 1, j] - sol[i - 1, j - 1]) / h + f(x[j], t[i - 1]))

    return sol


def pure_implicit_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)

    diag = 1 + c * tau / h
    off_diag = -c * tau / h

    A = np.diag(diag * np.ones(n)) + np.diag(off_diag * np.ones(n - 1), k = -1)

    for i in range(1, k + 1):
        b = np.zeros(n + 1)
        for j in range(1, n + 1):
            b[j] = sol[i - 1, j] + tau * f(x[j], t[i - 1])

        sol[i, 1:] = np.linalg.solve(A, b[1:])

    return sol


def implicit_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)

    diag = 1 + c * tau / h
    under_diag = -c * tau / h

    A = np.diag(diag * np.ones(n)) + np.diag(under_diag * np.ones(n - 1), k=-1)

    for i in range(1, k + 1):
        b = np.zeros(n + 1)
        for j in range(1, n + 1):
            b[j] = sol[i - 1, j - 1] + tau * f(x[j], t[i - 1])

        sol[i, 1:] = np.linalg.solve(A, b[1:])

    return sol


def symmetric_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)

    diag = 2 + 2 * c * tau / h
    under_diag = -c * tau / h

    A = np.diag(diag * np.ones(n)) + np.diag(under_diag * np.ones(n - 1), k=-1) + np.diag(under_diag * np.ones(n - 1),
                                                                                          k=1)

    for i in range(1, k + 1):
        b = np.zeros(n + 1)
        for j in range(1, n + 1):
            b[j] = 2 * sol[i - 1, j - 1] + 2 * tau * f((x[j] + h / 2), (t[i] + tau / 2))

        sol[i, 1:] = np.linalg.solve(A, b[1:])

    return sol