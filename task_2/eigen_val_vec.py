import numpy as np
from scipy.special import jacobi

from fields import *
from task_2.func_operations import find_scalar_product


# Вычисляем собственное значение при заданном k
def find_eigen_val(k, p, q):
    return np.pi**2*k**2/(x1-x0)**2*p+q


# Вычисляем собственный вектор для заданного k
def eigen_vec(k):
    ck = np.tan(np.pi/(x1-x0)*k)
    fun = lambda x: ck*np.cos(np.pi/(x1-x0)*k*x)+np.sin(np.pi/(x1-x0)*k*x)
    # Возвращаем собственный вектор, нормированный на единичную длину
    # Нормируем при помощи деления fun(x) на sqrt из скалярного произведения fun(x) на саму себя
    return lambda x: fun(x)/np.sqrt(find_scalar_product(fun, fun))


# Вычисляем производную собственного вектора для заданного k
def deigen_vec(k):
    ck = np.tan(np.pi/(x1-x0)*k)
    fun = lambda x: ck*np.cos(np.pi/(x1-x0)*k*x)+np.sin(np.pi/(x1-x0)*k*x)
    # Вычисляем норму собственной функции, чтобы найти длину вектора в пространстве функций
    c = np.sqrt(find_scalar_product(fun, fun))
    # Вычисляем значение a, используемое в производной собственной функции
    a = np.pi/(x1-x0)*k
    # Возвращаем производную собственной функции и нормализуем на значение c
    return lambda x: (a*np.cos(a*x)-ck*a*np.sin(a*x))/c


# Вычисляем вторую производную собственного вектора для заданного k
def ddeigen_vec(k):
    ck = np.tan(np.pi/(x1-x0)*k)
    fun = lambda x: ck*np.cos(np.pi/(x1-x0)*k*x)+np.sin(np.pi/(x1-x0)*k*x)
    c = np.sqrt(find_scalar_product(fun, fun))
    a = np.pi/(x1-x0)*k
    return lambda x: (-a*a*np.sin(a*x)-ck*a*a*np.cos(a*x))/c


# Используем jacobi для вычисления полиномов Якоби (нужны для создания базисных функций)
# Вычисляем базисные функции и нормируем их
def coordinate_func(k):
    fun = lambda x: (1-((2*x-x0-x1)/(x1-x0))**2)*jacobi(k, 2, 2)((2*x-x0-x1)/(x1-x0))
    c = np.sqrt(find_scalar_product(fun, fun))
    return lambda x: fun(x)/c


# Вычисляем производные базисных функций и нормируем их
# Если k=0, используем формулу для вычисления производной базисной функции 1-го порядка
# Иначе 2-го порядка
def d_coordinate_func(k):
    fun = lambda x: (1-((2*x-x0-x1)/(x1-x0))**2)*jacobi(k, 2, 2)((2*x-x0-x1)/(x1-x0))
    c = np.sqrt(find_scalar_product(fun, fun))
    if k == 0:
        return lambda x: (-4*(2*x-x0-x1)/(x1-x0)**2*jacobi(k, 2, 2)((2*x-x0-x1)/(x1-x0)))/c
    return lambda x: (-4*(2*x-x0-x1)/(x1-x0)**2*jacobi(k, 2, 2)((2*x-x0-x1)/(x1-x0)) +
                      2/(x1-x0)*(1-((2*x-x0-x1)/(x1-x0))**2) * (k+5)/2*jacobi(k-1, 3, 3)((2*x-x0-x1)/(x1-x0)))/c


# Создаем СФ на основе полученных в методе Галеркина коэффициентов
def eig_vec_r(n, coef):
    fun = lambda x: sum([coordinate_func(i)(x) * coef[i] for i in range(n)])
    return lambda x: fun(x)


# Метод обратных итераций для поиска минимального СЗ и соответствующего СВ матрицы Гамма
def reverse_iterations_method(Gamma_L, epsilon=1e-4):
    # Получаем размерность матрицы Гамма
    n = Gamma_L.shape[0]
    # Генерируем случайный начальный вектор z
    z = np.random.rand(n)
    # Нормируем вектор z

    z /= np.linalg.norm(z)
    while True:
        # Решаем систему линейных уравнений Гамма для нахождения нового вектора
        z_new = np.linalg.solve(Gamma_L, z)

        # Вычисляем норму нового вектора и нормируем его
        z_new_norm = np.linalg.norm(z_new)
        z_new /= z_new_norm

        # Если разница между предыдущим и новым векторами меньше эпсилон, то завершаем цикл
        if np.linalg.norm(z_new - z) < epsilon:
            break

        # Обновляем вектор z
        z = z_new

    # Вычисляем min СЗ
    lambda_min = np.dot(z, np.dot(Gamma_L, z)) / np.dot(z, z)
    # Возвращаем min СЗ и соответствующий СВ
    return lambda_min, z