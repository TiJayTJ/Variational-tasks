import numpy as np

from task_4.math_functions import u_0


def explicit_scheme(f, c, t_step, x_step, x_steps_number, t_steps_number, x, t):
    sol = np.zeros((t_steps_number + 1, x_steps_number + 1))
    sol[0, :] = u_0(x)
    sol[:, 0] = u_0(t)

    for i in range(1, t_steps_number + 1):
        for j in range(1, x_steps_number + 1):
            sol[i, j] = sol[i - 1, j] + t_step * (-c * (sol[i - 1, j] - sol[i - 1, j - 1]) / x_step + f(x[j], t[i - 1]))

    return sol


def pure_implicit_scheme(f, c, tau, h, n, k, x, t):
    sol = np.zeros((k + 1, n + 1))
    sol[0, :] = u_0(x)
    sol[:, 0] = u_0(t)

    diag = 1 + c * tau / h
    off_diag = -c * tau / h

    A = np.diag(diag * np.ones(n)) + np.diag(off_diag * np.ones(n - 1), k = -1)

    for i in range(1, k + 1):
        b = np.zeros(n + 1)
        for j in range(1, n + 1):
            b[j] = sol[i - 1, j] + tau * f(x[j], t[i - 1])

        sol[i, 1 : ] = np.linalg.solve(A, b[1 : ])

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

    diag = 1 + c * tau / h
    under_diag = 1 - c * tau / h

    A = np.diag(diag * np.ones(n)) + np.diag(under_diag * np.ones(n - 1), k=-1)

    for i in range(1, k + 1):
        b = np.zeros(n + 1)
        for j in range(1, n + 1):
            b[j] = diag * sol[i - 1, j - 1] + under_diag * sol[i - 1, j] + 2 * tau * f((x[j] + h / 2), (t[i] + tau / 2))

        sol[i, 1:] = np.linalg.solve(A, b[1:])

    return sol