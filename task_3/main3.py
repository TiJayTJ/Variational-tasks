from sympy import solve, series, sinh, symbols


import pandas as pd
import numpy as np

import warnings

from task_3.fields import *
from task_3.integral_equation_solving_methods import singular_kernel, mech_quad

warnings.filterwarnings("ignore")

print("Численное решение интегрального уравнения Фредгольма второго рода")
print("Вариант 4")

# Вычисляем ряд Тейлора для sh(xy) относительно y, начиная с y=0 и ограничиваемся N
# Удаляем члены более высокого порядка и проделываем то же самое относительно x
x, y = symbols('x y')

sinh_taylor = series(sinh(x*y), y, 0, N).removeO().series(x, 0, N).removeO()


# Выводим невязки
data = {'N': [], 'Невязка': []}

N_values = range(1, 10)

for N in N_values:
    _, loss = singular_kernel(N)
    data['N'].append(N)
    data['Невязка'].append(loss)

df = pd.DataFrame(data)
print(df)
print()


# Метод механических квадратур


# Приближенное решение уравнения (функция u(x))
u_true, _ = singular_kernel(40)


# Вычисляем ошибку между численным решением u и точным решением u_true, используя максимальное абсолютное отклонение
def loss(u, x):
    return np.max(np.abs(u - u_true(x)))


# Решаем уравнение интегрального типа на [0,1], используя 10 узлов сетки и параметр lambd
U = mech_quad(0, 1, 10, lambd)

# Численное решение для различных N и оценка ошибки для каждого решения
Ns = [11, 21, 41, 81, 101, 201]

results = pd.DataFrame(columns=['N', 'Оценка ошибки'])

# Вычисляем шаг сетки для текущего N; создаем равномерную сетку на [a,b]; численное решение уравнения;
# вычисляем оценку ошибки и выводим результат
for N in Ns:
    h = (b - a) / (N - 1)
    x = np.linspace(a, b, N)
    u = mech_quad(a, b, N, lambd)
    current_loss = loss(u, x)
    results = pd.concat([results, pd.DataFrame({'N': [N], 'Оценка ошибки': [current_loss]})], ignore_index=True)


print(results)