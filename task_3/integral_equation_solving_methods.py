import math
import numpy as np
from numpy.linalg import solve

from task_3.func_operations import find_scalar_product
from task_3.fields import *

# Метод замены ядра

# Решение уравнения интегрального типа методом сингулярного интегрального уравнения
def singular_kernel(N):
    # Создаем список анонимных функций An, которые представляют собой члены разложения в ряд
    # Тейлора функции x^k/k! для аппроксимации ядра интегрального уравнения
    An = [lambda x, k=i: x ** k / math.factorial(k) for i in range(1, 2 * N, 2)]
    # Создаем список анонимных функций Bn, которые представляют собой многочлены степени k для
    # аппроксимации функции правой части уравнения
    Bn = [lambda x, k=i: x ** k for i in range(1, 2 * N, 2)]

    # Нулевая матрица (матрица системы уравнений) и нулевой вектор (вектор правой части уравнения)
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Заполняем матрицу A и вектор b
    for i in range(N):
        for j in range(N):
            A[i, j] = delta(i, j) - lambd * find_scalar_product(Bn[i], An[j])

    for i in range(N):
        b[i] = find_scalar_product(Bn[i], f)

    # Решаем систему линейных уравнений, где an - вектор коэффициентов аппроксимации An(x)
    cn = np.linalg.solve(A, b)

    # u(x) - сумма функции правой части f(x) и линейной комбинации аппроксимацией An(x) с
    # найденными коэффициентами an
    u = lambda x: f(x) + lambd * sum([cn[i] * An[i](x) for i in range(N)])

    # Создаем массив точек на [0,1] для оценки u(x)
    x  = np.linspace(0, 1, 1000)

    # Вычисляем невязку между левой и правой частью уравнению для найденного приближенного решения u(x)
    def loss(u):
        left_side = lambda x: u(x) - lambd * find_scalar_product(lambda y: np.sinh(x * y), u)
        right_side = f
        left_side_val = np.array([left_side(xi) for xi in x])
        return np.max(np.abs(left_side_val - right_side(x)))

    # Возвращаем приближенное решение u(x) и функцию для оценки невязки
    return u, loss(u)


# Метод механических квадратур


# Определяем ядро уравнения интегрального типа
def K(x, y):
    return np.sinh(x * y)


# Численный метод для решения уравнения с использованием квадратурных формул
# Создаем сетку x, строим матрицу системы уравнений, решаем систему и возвращаем вектор решения
def mech_quad(a, b, N, lambd):
    x = np.linspace(a, b, N)
    h = (b - a) / (N - 1)
    # Создаем единичную матрицу
    A = np.eye(N)
    # Вычисляем значение функции правой части уравнения в узлах сетки
    F = f(x)

    # Заполняем матрицу A, учитывая ядро K(x,y) и параметр lambd. m - индекс столбца, n - индекс строки
    # m соответствует краевым узлам - квадратная формула с весом h/3
    # m нечетный - квадратная формула с весом 4h/3
    # m четный - квадратная формула с весом 2h/3
    for n in range(N):
        for m in range(N):
            if m == 0 or m == N - 1:
                cm = h / 3
            elif m % 2 == 0:
                cm = 2 * h / 3
            else:
                cm = 4 * h / 3
            A[n, m] -= lambd * cm * K(x[n], x[m])

    # Решаем систему линейных уравнений и возвращаем U - численное решение уравнения
    U = solve(A, F)
    return U