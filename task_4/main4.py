import numpy as np
import matplotlib.pyplot as plt

from task_4.math_functions import f
from task_4.schemes import *


def surface_drawing(t, x, sol):
    T, X = np.meshgrid(t, x)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, X, sol.T, cmap='viridis')

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u(t, x)')
    ax.set_title('Solution of the PDE')
    plt.colorbar(surf)
    plt.show()


# Явная схема


# Устойчивый случай

a = 1
T = 0.1
N = M = 100
h = tau = 1 / N
c = 1.0

if c * tau / h <= 1:
    print("Устойчивая")
else:
    print("Неустойчивая")
print(c * tau / h)

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, M + 1)

u = explicit_scheme(f, c, tau, h, N, M, x, t)

surface_drawing(t, x, u)
plt.plot(x, u[40].T)


# Неустойчивый случай

a = 1
T = 1
N = 700
M = 100
h = 1 / N
tau = 1 / M
c = 1.0

if (c * tau / h <= 1):
    print("Устойчивая")
else:
    print("Неустойчивая")
print(c * tau / h)

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, M + 1)

u = explicit_scheme(f, c, tau, h, N, M, x, t)

surface_drawing(t, x, u)
plt.plot(x, u[6].T)


# Чисто неявная схема
# (всегда устойчива)

a = 10
T = 3
N = 100
K = 200
h = a / N
tau = T / K
c = 1.0


x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, K + 1)

sol = pure_implicit_scheme(f, c, tau, h, N, K, x, t)

surface_drawing(t, x, sol)
plt.plot(x, sol[15].T)


# Неявная схема


# Неустойчивый случай

a = 10
T = 3
N = 200
K = 100
h = a / N
tau = T / K
c = 1.0

if (c * tau / h >= 1):
    print("Устойчивая")
else:
    print("Неустойчивая")
print(c * tau / h)

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, K + 1)

sol = implicit_scheme(f, c, tau, h, N, K, x, t)

surface_drawing(t, x, sol)
plt.plot(x, sol[6].T)

# Устойчивый случай

a = 10
T = 3
N = 1000
K = 100
h = a / N
tau = T / K
c = 1.0

if (c * tau / h >= 1):
    print("Устойчивая")
else:
    print("Неустойчивая")
print(c * tau / h)

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, K + 1)

sol = implicit_scheme(f, c, tau, h, N, K, x, t)

surface_drawing(t, x, sol)
plt.plot(x, sol[8].T)


# Симметричная схема
# (всегда устойчива)


a = 10
T = 3
N = 100
K = 200
h = a / N
tau = T / K
c = 1.0

x = np.linspace(0, a, N + 1)
t = np.linspace(0, T, K + 1)

sol = symmetric_scheme(f, c, tau, h, N, K, x, t)

surface_drawing(t, x, sol)
plt.plot(x, sol[15].T)