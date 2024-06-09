import numpy as np
import sympy

from task_1.maths import get_coordinate_func, get_coordinate_func_values, get_cheb_nodes


# Метод Ритца
def run_ritz_method(k, n):
    # Вызываем функция Якоби и метода координатых функций и их производных
    phis, _, _ = get_coordinate_func(k, n)

    # Создаем матрицу и вектор для построения системы линейных уравнений метода Ритца
    matrix_a = np.zeros((n, n))
    vector_b = np.zeros((n, 1))

    var_x = sympy.symbols('x')
    # Определяем функцию f(x)
    f = 2 - var_x

    # 1) Вычисляем произведение f(x) на одну из координатных функций
    # 2) Вычисляем значение интеграла от h на [-1,1] (элементы вектора правой части)
    for i in range(3):
        h = f * phis[i]
        vector_b[i] = sympy.integrals.integrate(h, (var_x, -1, 1))

    # Задаем значения узлов и соответствующие веса для формулы Гаусса
    x1 = 1 / 3 * (5 - 2 * (10 / 7) ** 0.5) ** 0.5
    x2 = 1 / 3 * (5 + 2 * (10 / 7) ** 0.5) ** 0.5
    c1 = (322 + 13 * 70 ** 0.5) / 900
    c2 = (322 - 13 * 70 ** 0.5) / 900
    x_i = [-x2, -x1, 0, x1, x2]
    c_i = [c2, c1, 128 / 225, c1, c2]

    # Вычисляем значения координатных функция и их производных в этом узле
    phis_values, d_phis_values, dd_phis_values = get_coordinate_func_values(k, n, x_i)

    # Вычисление интеграла по формуле Гаусса
    def find_gauss_integral(nodes, coefs, i, j):
        s = 0
        # Перебираем все узлы формулы Гаусса
        for k in range(len(nodes)):
            tmp_1 = (((nodes[k] + 4) / (nodes[k] + 5)) * d_phis_values[k][j] * d_phis_values[k][i] + np.exp(
                nodes[k] ** 4 / 4) *
                     phis_values[k][i] * phis_values[k][j])
            s += coefs[k] * tmp_1
        return s

    for i in range(n):
        for j in range(n):
            matrix_a[i][j] = find_gauss_integral(x_i, c_i, i, j)

    # Вычисляем коэффициенты для решения методом Ритца
    coeffs = np.linalg.solve(matrix_a, vector_b)
    return coeffs, matrix_a, vector_b


def run_collocation_method(k, n):
    # Генерируем узлы метода коллокации с помощью Чебышева
    nodes = get_cheb_nodes(n)
    # Функции, которые представляют правую часть дифура и коэффициенты при производных в уравнении
    f = lambda x: 2 - x
    p = lambda x: (x + 4) / (x + 5)
    dp = lambda x: 1 / (x + 5) ** 2
    r = lambda x: np.exp(x ** 4 / 4)

    # Матрица и вектор для построения системы линейных уравнений метода коллокации
    matrix_a = np.zeros((n, n))
    vector_b = np.zeros((n, 1))

    # 1) Вычисляем значение правой части дифура в узле
    # 2) Вычисляем значения координатных функций и их производных в узле
    for i in range(n):
        vector_b[i] = f(nodes[i])
        # Вычисляем значения координатных функция и их производных в этом узле
        phis_values, d_phis_values, dd_phis_values = get_coordinate_func_values(k, n, nodes)
        for j in range(n):
            # Вычисляем значения произведения и второй, первой производной и координатной функции в узле
            tmp1 = p(nodes[i]) * dd_phis_values[i][j]
            tmp2 = dp(nodes[i]) * d_phis_values[i][j]
            tmp3 = r(nodes[i]) * phis_values[i][j]
            matrix_a[i][j] = (-1) * (tmp1 + tmp2) + tmp3
    # Решаем систему линейных уравнений, чтобы найти коэффициенты приближенного решения
    coeffs = np.linalg.solve(matrix_a, vector_b)
    return coeffs, matrix_a, vector_b