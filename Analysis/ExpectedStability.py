import numpy as np
from init import *


def stability(reps, r):
    s = start_stability
    for _ in range(reps):
        s *= 1.5 * a * np.power(s, -b) * np.log(r) + 1
    return s


def estability(reps, r):
    es = start_stability
    for k in range(reps):
        es = np.power(r, k + 1) * stability(k + 1, r) + (1 - r) * es
    return es


def matrix():
    estability_matrix = np.empty(shape=[10, 9])
    for i in range(1, 11):
        for j in range(1, 10):
            reps = i
            r = j / 10
            estability_matrix[i - 1][j - 1] = estability(reps, r)
    print(estability_matrix)


if __name__ == "__main__":
    matrix()
