# Строит массив многочленов Якоби
import numpy as np
import sympy
from sympy.abc import x


def get_jacobi_poly(k, n):
    pj = [x] * (n + 1)
    pj[0] = 1
    pj[1] = (1 + k) * x
    for j in range(2, n + 1):
        tmp_3 = (j + 2 * k) * j
        tmp_1 = (j + k) * (2 * (j - 2) + 2 * k + 3)
        tmp_2 = (j + k) * (j + k - 1)
        pj[j] = (tmp_1 * x * pj[j - 1] - tmp_2 * pj[j - 2]) / tmp_3
    return pj


# Строит массив значение многочленов Якоби в точке y
def get_jacobi_poly_values(functions, y):
    vals = [float(substitute_if_expr(p, y)) for p in functions]
    return vals


# Функция для подстановки значения только если объект является выражением SymPy
def substitute_if_expr(p, value):
    if hasattr(p, 'subs'):
        return p.subs(x, value)
    else:
        return "Substitution error"


# Строит массивы координатных функций и их производных
def get_coordinate_func(k, n):
    phi = [x] * n
    d_phi = [x] * n
    dd_phi = [x] * n

    jac_polys = get_jacobi_poly(k, n)
    d_jac_polys = get_jacobi_poly(k - 1, n + 1)
    for i in range(n):
        phi[i] = (1 - x ** 2) * jac_polys[i]
        phi[i] = sympy.simplify(phi[i])

        d_phi[i] = (-2) * (i + 1) * (1 - x ** 2) ** (k - 1) * d_jac_polys[i + 1]
        d_phi[i] = sympy.simplify(d_phi[i])

        tmp1 = (k - 1) * (1 - x ** 2) ** (k - 2) * d_jac_polys[i + 1]
        tmp2 = (1 - x ** 2) ** (k - 1) * ((i + 1 + 2 * (k - 1) + 1) / 2) * jac_polys[i]
        dd_phi[i] = (-2) * (i + 1) * (tmp1 + tmp2)
        dd_phi[i] = sympy.simplify(dd_phi[i])

    return phi, d_phi, dd_phi


def get_coordinate_func_values(k, n, x_i):
    phis, d_phis, dd_phis = get_coordinate_func(k, n)
    phis_values = [get_jacobi_poly_values(phis, x_i[i]) for i in range(len(x_i))]
    d_phis_values = [get_jacobi_poly_values(d_phis, x_i[i]) for i in range(len(x_i))]
    dd_phis_values = [get_jacobi_poly_values(dd_phis, x_i[i]) for i in range(len(x_i))]
    return phis_values, d_phis_values, dd_phis_values


# Строит узлы многочлена Чебышева первого рода
def get_cheb_nodes(n):
    arr = []
    for i in range(1, n + 1):
        arr.append(np.cos((2 * i - 1) * np.pi / (2 * n)))
    return arr