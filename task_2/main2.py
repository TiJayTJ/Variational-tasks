import numpy as np
import pandas as pd
from numpy.linalg import eig

import matplotlib.pyplot as plt

from task_2.eigen_val_vec import eigen_vec, find_eigen_val, deigen_vec, ddeigen_vec, coordinate_func, eig_vec_r, \
    d_coordinate_func, reverse_iterations_method
from task_2.func_operations import find_energy_scalar_product, find_l2_scalar_product
from task_2.math_functions import p, q
from task_2.fields import *

print("Проблема собственных значений в задаче Штурма-Лиувилля")
print("Вариант 4")


# Вычисление собственного числа и вектора по формуле


# Оценка функций p(x), q(x) (оцениваем диапазоны значений p(x) и q(x) на [x_0, x_1])
x = np.linspace(x0, x1, 1000)

# Вычисляем min и max функции p(x), построив массив значений p(x)
p_min = np.min(p(np.linspace(x0, x1, 1000)))
p_max = np.max(p(np.linspace(x0, x1, 1000)))

# Вычисляем min и max функции q(x), построив массив значений q(x)
q_min = np.min(q(np.linspace(x0, x1, 1000)))
q_max = np.max(q(np.linspace(x0, x1, 1000)))


for i in range(2):
    plt.plot(x, eigen_vec(i+1)(x), label=f'Собственная функция {i+1}')
plt.legend()
plt.show()


# Вычисляем невязку между левой и правой частями уравнения для СФ
def loss(i, pm, qm, x):
    # Определяем левую часть уравнения для СФ
    left_side = lambda x: -k*deigen_vec(i)(x)-p(x)*ddeigen_vec(i)(x)+q(x)*eigen_vec(i)(x)
    # Определяем правую часть уравнения для СФ
    right_side = lambda x: find_eigen_val(i, pm, qm) * eigen_vec(i)(x)
    # Находим max абсолютное значение разности между левой и правой частями
    # Оцениваем, насколько близки эти части (чем меньше значение, тем ближе СФ к удовлетворяющей уравнению форме)
    return np.max(np.abs(right_side(x)-left_side(x)))


# Таблица с результатами вычислений СЗ и их невязки для 1-ых двух СФ при разных p и q
df_mm = pd.DataFrame()

df_mm['p'] = ['min', 'max']
df_mm['lambda_1 оценки'] = [find_eigen_val(1, p_min, q_min), find_eigen_val(1, p_max, q_max)]
df_mm['lambda_1 невязки'] = [loss(1, p_min, q_min, x), loss(1, p_max, q_max, x)]

df_mm['lambda_2 оценки'] = [find_eigen_val(2, p_min, q_min), find_eigen_val(2, p_max, q_max)]
df_mm['lambda_2 невязки'] = [loss(2, p_min, q_min, x), loss(2, p_max, q_max, x)]

print(df_mm)

# Вычисляем собственные числа через "точные" собственные функции
print(f'Первое точное собственное число {find_energy_scalar_product(eigen_vec(1), eigen_vec(1), 
                                                             deigen_vec(1), deigen_vec(1)) / 
                                  find_l2_scalar_product(eigen_vec(1), eigen_vec(1))}')
print(f'Второе точное собственное число {find_energy_scalar_product(eigen_vec(2), eigen_vec(2),
                                                             deigen_vec(2), deigen_vec(2)) / 
                                  find_l2_scalar_product(eigen_vec(2), eigen_vec(2))}')


# Метод Ритца


# Создаем матрицу Грама и вычисляем ее элементы
G_l = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        G_l[i, j] = find_energy_scalar_product(coordinate_func(i), coordinate_func(j), d_coordinate_func(i), d_coordinate_func(j))


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


for i in range(2):
    plt.plot(x, eig_vec_r(N, vecs[:, i])(x), label=f'Собственная функция {i+1}')
plt.legend()
plt.show()


# Метод обратных итераций


# Задаем min СЗ матрицы Грама и соответствующий СВ
lambd, coefs = reverse_iterations_method(G_l)

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
                G_l[i, j] = find_energy_scalar_product(coordinate_func(i), coordinate_func(j), d_coordinate_func(i), d_coordinate_func(j))
        lambd, coefs = reverse_iterations_method(G_l)
        lambd_list.append(lambd)
        lambd_loss.append(lambd-vals[0])
    df['lambda_1^{(n)}'] = lambd_list
    df['lambda_1^{(n)}-lambda_1^*'] = lambd_loss

    return df


print(table([3, 4, 5, 6, 7]))


