import numpy as np
from scipy.special import jacobi

from task_2.fields import *


# Определяем коэффициенты перед u
def p(x):
    return k * x + l


def q(x):
    return k**2 / (k * x + l) - k**3 * x

