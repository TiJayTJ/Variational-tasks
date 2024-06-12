import numpy as np

from fields import *
from task_2.func_operations import find_l2_scalar_product


# Вычисляем собственное значение при заданном k
def find_eigen_val(k, p, q):
    return np.pi**2*k**2/(x1-x0)**2*p+q


# Вычисляем собственный вектор для заданного k
def eigen_vec(k):
    ck = np.tan(np.pi/(x1-x0)*k)
    fun = lambda x: ck*np.cos(np.pi/(x1-x0)*k*x)+np.sin(np.pi/(x1-x0)*k*x)
    # Возвращаем собственный вектор, нормированный на единичную длину
    # Нормируем при помощи деления fun(x) на sqrt из скалярного произведения fun(x) на саму себя, используя dotL2
    return lambda x: fun(x)/np.sqrt(find_l2_scalar_product(fun, fun))


# Вычисляем производную собственного вектора для заданного k
def deigen_vec(k):
    ck = np.tan(np.pi/(x1-x0)*k)
    fun = lambda x: ck*np.cos(np.pi/(x1-x0)*k*x)+np.sin(np.pi/(x1-x0)*k*x)
    # Вычисляем норму собственной функции, чтобы найти длину вектора в пространстве функций
    c = np.sqrt(find_l2_scalar_product(fun, fun))
    # Вычисляем значение a, используемое в производной собственной функции
    a = np.pi/(x1-x0)*k
    # Возвращаем производную собственной функции и нормализуем на значение c
    return lambda x: (a*np.cos(a*x)-ck*a*np.sin(a*x))/c


# Вычисляем вторую производную собственного вектора для заданного k
def ddeigen_vec(k):
    ck = np.tan(np.pi/(x1-x0)*k)
    fun = lambda x: ck*np.cos(np.pi/(x1-x0)*k*x)+np.sin(np.pi/(x1-x0)*k*x)
    c = np.sqrt(find_l2_scalar_product(fun, fun))
    a = np.pi/(x1-x0)*k
    return lambda x: (-a*a*np.sin(a*x)-ck*a*a*np.cos(a*x))/c