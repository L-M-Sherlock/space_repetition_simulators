import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from init import *
import math


def sinc(s, r):
    return a * np.power(s, -b) * np.log(r) + 1


def expected_s(s, r):
    return r * s * sinc(s, r) + (1 - r) * start_stability


def stress(s):
    # return - np.log(s)
    # return 1 / s
    # return - s
    return - np.power(s, 1 / 2)


def expected_stress(s, r):
    return r * stress(s * sinc(s, r)) + (1 - r) * stress(start_stability) \
           + np.log(0.9) / np.log(r)


def opt_s(r):
    return - expected_s(1, r)


def opt_stress(r):
    return expected_stress(1, r)


def diff_recall():
    num_samples = 10000
    day_long = 2000
    recalls = [0.9, 0.8, 0.6, 0.2]
    for recall in recalls:
        repeats = np.array([0.0 for _ in range(day_long)])
        for _ in range(num_samples):
            day = 0
            s = 1
            while day < day_long:
                ivl = round(s)
                for i in range(ivl):
                    if i + day >= day_long:
                        break
                    repeats[day + i] += 1 / ivl
                if random.random() < recall:
                    s *= a * np.power(s, -b) * np.log(recall) + 1
                else:
                    s = np.log(recall) / np.log(0.9)
                day += ivl
        repeats = repeats / num_samples
        for i in range(day_long - 1):
            repeats[i + 1] += repeats[i]
        plt.plot(repeats, "-", label=f'保留率{recall}')
    plt.grid(True)
    plt.xlabel("时间/天")
    plt.legend()
    plt.show()


def diff_start():
    num_samples = 10000
    day_long = 3000
    starts = [1, 2, 4, 8]
    recall = 0.9
    repeats = []
    for start in starts:
        repeats.append(np.array([0.0 for _ in range(day_long)]))
        for _ in range(num_samples):
            day = 0
            s = start
            while day < day_long:
                ivl = round(s)
                day += ivl
                for i in range(ivl):
                    if day >= day_long:
                        break
                    repeats[-1][day - i - 1] += 1 / ivl
                if random.random() < recall:
                    s *= a * np.power(s, -b) * np.log(recall) + 1
                else:
                    s = np.log(recall) / np.log(0.9)
        repeats[-1] = repeats[-1] / num_samples
        for i in range(day_long - 1):
            repeats[-1][i + 1] += repeats[-1][i]
        plt.figure(1)
        plt.plot(repeats[-1], "-", label=f'初始稳定性{start}')
    plt.grid(True)
    plt.xlabel("时间/天")
    plt.legend()
    plt.figure(2)
    plt.grid(True)
    plt.xlabel("时间/天")
    for i in range(len(starts) - 1):
        result = repeats[0] - repeats[i+1]
        mean = sum(result) / len(result)
        plt.plot(result, label=f'初始稳定性{starts[i+1]}-均值{mean:.2f}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    r1 = optimize.fminbound(opt_s, 0, 1)
    print(r1)
    r2 = optimize.fminbound(opt_stress, 0, 1)
    print(r2)
    # diff_recall()
    diff_start()
