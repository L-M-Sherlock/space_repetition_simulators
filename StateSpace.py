import numpy as np
from init import *


def cal_SInc_log(S, R):
    return a * np.power(S, -b) * np.log(R) + 1


def cal_SInc_exp(S, R):
    return - a * np.power(S, -b) * (np.exp(1 - R) - 1) * 1.1 + 1


def cal_SInc_linear(S, R):
    return - a * np.power(S, -b) * (1 - R) * 1.4 + 1


if __name__ == "__main__":
    sinc_log_matrix = np.empty(shape=[10, 10])
    esinc_log_matrix = np.empty(shape=[10, 10])
    for i in range(1, 11):
        for j in range(1, 11):
            s = np.power(2, j)
            r = i / 10
            sinc_log_matrix[i - 1][j - 1] = cal_SInc_log(s, r)
            esinc_log_matrix[i - 1][j - 1] = sinc_log_matrix[i - 1][j - 1] * r + 1 / s * (1 - r)
    sinc_exp_matrix = np.empty(shape=[10, 10])
    esinc_exp_matrix = np.empty(shape=[10, 10])
    for i in range(1, 11):
        for j in range(1, 11):
            s = np.power(2, j)
            r = i / 10
            sinc_exp_matrix[i - 1][j - 1] = cal_SInc_exp(s, r)
            esinc_exp_matrix[i - 1][j - 1] = sinc_exp_matrix[i - 1][j - 1] * r + 1 / s * (1 - r)
    sinc_linear_matrix = np.empty(shape=[10, 10])
    esinc_linear_matrix = np.empty(shape=[10, 10])
    for i in range(1, 11):
        for j in range(1, 11):
            s = np.power(2, j)
            r = i / 10
            sinc_linear_matrix[i - 1][j - 1] = cal_SInc_linear(s, r)
            esinc_linear_matrix[i - 1][j - 1] = sinc_linear_matrix[i - 1][j - 1] * r + 1 / s * (1 - r)
    print("end")
