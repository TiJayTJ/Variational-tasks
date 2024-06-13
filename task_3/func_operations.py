import scipy.integrate as integrate


# Вычисляем скалярное произведение f1(x) и f2(x) на [0,1] путем численного интегрирования
def dotL2(f1,f2):
    return integrate.quad(lambda x: f1(x)*f2(x), 0, 1)[0]


# Вычисляем дискретное скалярное произведение функции k(x,y) и вектора u на сетке x с шагом h
def dotL2dis(k, u, x, h):
    integral_approx = 0
    for i, xi in enumerate(x):
        integral_approx += k(xi, x) * u[i] * h
    return integral_approx