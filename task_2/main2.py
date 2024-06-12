import numpy as np
import pandas as pd
from scipy.special import jacobi
from numpy.linalg import eig

import matplotlib.pyplot as plt

from task_2.eigen_val_vec import eigen_vec, find_eigen_val, deigen_vec, ddeigen_vec
from task_2.func_operations import find_energy_scalar_product, find_l2_scalar_product
from task_2.math_functions import p, q
from task_2.fields import *

# Оценка функций p(x), q(x) (оцениваем диапазоны значений p(x) и q(x) на [x_0, x_1])
x = np.linspace(x0, x1, 1000)

# Вычисляем min и max функции p(x), построив массив значений p(x)
p_min = np.min(p(np.linspace(x0, x1, 1000)))
p_max = np.max(p(np.linspace(x0, x1, 1000)))

# Вычисляем min и max функции q(x), построив массив значений q(x)
q_min = np.min(q(np.linspace(x0, x1, 1000)))
q_max = np.max(q(np.linspace(x0, x1, 1000)))


# Используем jacobi для вычисления полиномов Якоби (нужны для создания базисных функций)
# Вычисляем базисные функции и нормируем их
def coordinate_func(k):
    fun = lambda x: (1-((2*x-x0-x1)/(x1-x0))**2)*jacobi(k, 2, 2)((2*x-x0-x1)/(x1-x0))
    c = np.sqrt(find_l2_scalar_product(fun, fun))
    return lambda x: fun(x)/c



for i in range(2):
    plt.plot(x, eigen_vec(i+1)(x), label=f'Собственная функция {i+1}')
plt.legend()
plt.show()


# Вычисляем невязку между левой и правой частями уравнения для СФ
def loss(i, pm, x):
    # Определяем левую часть уравнения для СФ
    left_side = lambda x: -k*deigen_vec(i)(x)-p(x)*ddeigen_vec(i)(x)+q(x)*eigen_vec(i)(x)
    # Определяем правую часть уравнения для СФ
    right_side = lambda x: find_eigen_val(i, pm, pm) * eigen_vec(i)(x)
    # Находим max абсолютное значение разности между левой и правой частями
    # Оцениваем, насколько близки эти части (чем меньше значение, тем ближе СФ к удовлетворяющей уравнению форме)
    return np.max(np.abs(right_side(x)-left_side(x)))


# Таблица с результатами вычислений СЗ и их невязки для 1-ых двух СФ при разных p и q
df_mm = pd.DataFrame()

df_mm['p'] = ['min', 'max']
df_mm['lambda_1 оценки'] = [find_eigen_val(1, p_min, q_min), find_eigen_val(1, p_max, q_max)]
df_mm['lambda_1 невязки'] = [loss(1, p_min, q_min), loss(1, p_max, q_max)]

df_mm['lambda_2 оценки'] = [find_eigen_val(2, p_min, q_min), find_eigen_val(2, p_max, q_max)]
df_mm['lambda_2 невязки'] = [loss(2, p_min, q_min), loss(2, p_max, q_max)]

print(df_mm)

# Вычисляем собственные числа через "точные" собственные функции
print(f'Первое собственное число {find_energy_scalar_product(eigen_vec(1), eigen_vec(1), 
                                                             deigen_vec(1), deigen_vec(1)) / 
                                  find_l2_scalar_product(eigen_vec(1), eigen_vec(1))}')
print(f'Второе собственное число {find_energy_scalar_product(eigen_vec(2), eigen_vec(2),
                                                             deigen_vec(2), deigen_vec(2)) / 
                                  find_l2_scalar_product(eigen_vec(2), eigen_vec(2))}')


# Метод Ритца





# Вычисляем производные базисных функций и нормируем их
# Если k=0, используем формулу для вычисления производной базисной функции 1-го порядка
# Иначе 2-го порядка
def dbasic_func(k):
    fun = lambda x: (1-((2*x-x0-x1)/(x1-x0))**2)*jacobi(k, 2, 2)((2*x-x0-x1)/(x1-x0))
    c = np.sqrt(find_l2_scalar_product(fun, fun))
    if k == 0:
        return lambda x: (-4*(2*x-x0-x1)/(x1-x0)**2*jacobi(k, 2, 2)((2*x-x0-x1)/(x1-x0)))/c
    return lambda x: (-4*(2*x-x0-x1)/(x1-x0)**2*jacobi(k, 2, 2)((2*x-x0-x1)/(x1-x0)) +
                      2/(x1-x0)*(1-((2*x-x0-x1)/(x1-x0))**2) * (k+5)/2*jacobi(k-1, 3, 3)((2*x-x0-x1)/(x1-x0)))/c


# Создаем матрицу Галеркина и вычисляем ее элементы
G_l = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        G_l[i, j] = find_energy_scalar_product(coordinate_func(i), coordinate_func(j), dbasic_func(i), dbasic_func(j))


# Вычисляем СЗ и СВ матрицы
vals, vecs = eig(G_l)
# Возвращаем индексы элементов массива vals в порядке возрастания
sorted_indices = np.argsort(vals)
# Упорядочиваем СЗ в порядке возрастания
vals = vals[sorted_indices]
# Упорядочиваем СВ в соответствии с отсортированными индексами
# Каждый столбец матрицы vecs соответствует СВ и переупорядочивается так, чтобы они
# соответствовали новому порядку СЗ
vecs = vecs[:, sorted_indices]


print(f'Первое собственное число {vals[0]}')
print(f'Второе собственное число {vals[1]}')


# Создаем СФ на основе полученных в методе Галеркина коэффициентов
def eig_vec_r(n, coef):
    fun = lambda x: sum([coordinate_func(i)(x) * coef[i] for i in range(n)])
    return lambda x: fun(x)


for i in range(2):
    plt.plot(x, eig_vec_r(N, vecs[:, i])(x), label=f'Собственная функция {i+1}')
plt.legend()
plt.show()


# Метод минимальных невязок для поиска минимального СЗ и соответствующего СВ матрицы Гамма
def scalar_product_method(Gamma_L, epsilon=1e-4):
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


# Задаем min СЗ матрицы Галеркина и соответствующий СВ
lambd, coefs = scalar_product_method(G_l)

plt.plot(x, sum([coordinate_func(i)(x) * coefs[i] for i in range(N)]), label=f'Собственная функция')
plt.legend()
plt.show()


# Таблица сравнения найденного min СЗ с точным значением
def table(Nn):
    df = pd.DataFrame()
    df['n'] = np.array(Nn)
    lambd_list = []
    lambd_loss = []

    for n in Nn:
        G_l = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                G_l[i, j] = find_energy_scalar_product(coordinate_func(i), coordinate_func(j), dbasic_func(i), dbasic_func(j))
        lambd, coefs = scalar_product_method(G_l)
        lambd_list.append(lambd)
        lambd_loss.append(lambd-vals[0])
    df['lambda_1^{(n)}'] = lambd_list
    df['lambda_1^{(n)}-lambda_1^*'] = lambd_loss

    return df


print(table([3, 4, 5, 6, 7]))


