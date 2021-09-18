from init import a, b
import numpy as np


def stability_inc_log(s, r):
    return (a * np.power(s, -b) * np.log(r)) + 1


def stability_inc_linear(s, r):
    return (a * np.power(s, -b) * (r - 1)) + 1


def stability_inc_exp(s, r):
    return (a * np.power(s, -b) * (- np.exp(1 - r) + 1)) + 1
