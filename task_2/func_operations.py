import numpy as np
import scipy.integrate as integrate

from task_2.fields import *
from task_2.math_functions import p, q


# Вычисляем численную производную вектора vec по x
def find_derivative(vec, dx):
    dvec = np.zeros_like(vec)
    # Для внутренних элементов вычисляем значения производных методом центральной разности
    # Производная вычисляется как разность значений соседних элементов vec / на двойной шаг dx
    dvec[1:-1] = (vec[2:] - vec[:-2]) / (2*dx)
    # Для 1-го элемента используем одностороннюю разность
    # Разности между 2-ым и 1-ым элементами vec / на шаг dx
    dvec[0] = (vec[1] - vec[0]) / dx
    # Для последнего: разность между последним и предпоследним элементами vec / на шаг dx
    dvec[-1] = (vec[-1] - vec[-2]) / dx
    return dvec


# Вычисляем L^2 норму вектора vec
# Интегрируем квадрат значений вектора с использованием метода трапеций и берем sqrt из результата
def find_l2_norm_integral(vec, dx):
    return np.sqrt(np.trapz(vec**2, dx=dx))


# Вычисляем скалярное произведение f1(x) и f2(x) на [x_0, x_1], используя quad для численного интегрирования
def find_l2_scalar_product(f1, f2):
    return integrate.quad(lambda x: f1(x)*f2(x), x0, x1)[0]


# Вычисляем интеграл с весом для пары функций w1(x) и w2(x) с их производными dw1(x), dw2(x)
# а также весами p(x), q(x), используя quad
def find_energy_scalar_product(w1, w2, dw1, dw2):
    return integrate.quad(lambda x: p(x)*dw1(x)*dw2(x)+q(x)*w1(x)*w2(x), x0, x1)[0]


# Вычисляем производную функции fun(x) в точке x, используя метод центральной разности с шагом h
def d_fun(fun, x):
    h = 1e-5
    return (fun(x+h)-fun(x-h))/(2*h)