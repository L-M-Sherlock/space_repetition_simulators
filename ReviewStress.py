import numpy as np
from scipy import optimize

a = -16
b = 0.23
start_stability = 1


def sinc(s, r):
    return a * np.power(s, -b) * np.log(r) + 1


def expected_s(s, r):
    return r * s * sinc(s, r) + (1 - r) * start_stability


def stress(s):
    return - np.log(s)
    # return 1 / s


def expected_stress(s, r):
    return r * stress(s * sinc(s, r)) + (1 - r) * stress(start_stability)


def opt_s(r):
    return - expected_s(10, r)


def opt_stress(r):
    return expected_stress(10, r)


if __name__ == "__main__":
    r1 = optimize.fminbound(opt_s, 0, 1)
    r2 = optimize.fminbound(opt_stress, 0, 1)
    print(r1)
    print(r2)
