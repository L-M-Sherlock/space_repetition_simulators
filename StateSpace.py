import numpy as np

a = -16
b = 0.23


def cal_SInc_log(S, R):
    return a * np.power(S, -b) * np.log(R) + 1


def cal_SInc_exp(S, R):
    return - a * np.power(S, -b) * (np.exp(1 - R) - 1) * 1.1 + 1


def cal_SInc_linear(S, R):
    return - a * np.power(S, -b) * (1-R) * 1.4 + 1


if __name__ == "__main__":
    sinc_log_matrix = np.empty(shape=[10, 10])
    for i in range(1, 11):
        for j in range(1, 11):
            sinc_log_matrix[i - 1][j - 1] = cal_SInc_log(np.power(2, j), i / 10)
    sinc_exp_matrix = np.empty(shape=[10, 10])
    for i in range(1, 11):
        for j in range(1, 11):
            sinc_exp_matrix[i - 1][j - 1] = cal_SInc_exp(np.power(2, j), i / 10)
    sinc_linear_matrix = np.empty(shape=[10, 10])
    for i in range(1, 11):
        for j in range(1, 11):
            sinc_linear_matrix[i - 1][j - 1] = cal_SInc_linear(np.power(2, j), i / 10)

    print("end")
