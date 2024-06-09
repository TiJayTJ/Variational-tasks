import numpy as np
import sympy
import pandas as pd

from task_1.maths import get_coordinate_func_values
from task_1.projection_methods import run_ritz_method, run_collocation_method


# Сравнивает с точным решением
def final_solution(coeffs, dots):
    exact_value = [0.721373, 0.813764, 0.541390]
    res = [0.0] * 3
    n = len(coeffs)

    # Вычисляем значения координатных функция и их производных в этом узле
    phis_values, _, _ = get_coordinate_func_values(1, n, dots)

    for i in range(3):
        for j in range(3):
            res[j] += coeffs[i] * phis_values[j][i]

    errs = [exact_value[k] - res[k] for k in range(3)]
    arr = [np.round(res[0], 5),
           np.round(res[1], 5),
           np.round(res[2], 5),
           np.round(errs[0], 5),
           np.round(errs[1], 5),
           np.round(errs[2], 5)]
    return arr


# Строит таблицу по значениям
def make_table(values):
    column = [
        "y(-0.5)",
        "y(0)",
        "y(0.5)",
        "y* - y(-0.5)",
        "y* - y(0)",
        "y* - y(0.5)"
    ]
    indexes = [3, 4, 5, 6, 7]
    table = pd.DataFrame(data=values, columns=column, index=indexes)
    table.columns.name = "n"
    return table


# Строим таблицу результатов для метода Ритца
dots = [-0.5, 0.0, 0.5]
val_Ritz = []
coefficients, A, b = [], [], []
for i in range(3, 8):
    coefficients, A, b = run_ritz_method(1, i)
    val_Ritz.append(final_solution(coefficients, dots))
result_table = make_table(val_Ritz)
print("Метод Ритца")
print("Расширенная матрица системы:")
print("А = ", A)

print("Число обусловленности матрицы А = ", np.linalg.cond(A))
print("b = ", b)
print("Коэффициенты разложения приближенного решения по координатным функциям:\n", coefficients)
print(result_table)

# Строим таблицу результатов для метода коллокации
dots = [-0.5, 0.0, 0.5]
val_colloc = []
coefficients, A, b = [], [], []
for i in range(3, 8):
    coefficients, A, b = run_collocation_method(1, i)
    val_colloc.append(final_solution(coefficients, dots))
result_table = make_table(val_colloc)
print("Метод коллокации")
print("Расширенная матрица системы:")
print("А = ", A)
print("Число обусловленности матрицы А = ", np.linalg.cond(A))
print("b = ", b)
print("Коэффициенты разложения приближенного решения по координатным функциям:\n", coefficients)
#print(result_table)
