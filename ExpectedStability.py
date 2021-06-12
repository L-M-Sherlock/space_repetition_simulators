import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (8.0, 4.0)
plt.rcParams['figure.dpi'] = 300

start_stability = 1
a = -16
b = 0.23


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


def diff_recall():
    num_samples = 10000
    day_long = 1000
    recalls = [0.95, 0.9, 0.8, 0.6, 0.2]
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
    day_long = 1000
    starts = [1, 10, 100, 1000]
    recall = 0.9
    for start in starts:
        repeats = np.array([0.0 for _ in range(day_long)])
        for _ in range(num_samples):
            day = 0
            s = start
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
        plt.plot(repeats, "-", label=f'初始稳定性{start}')
    plt.grid(True)
    plt.xlabel("时间/天")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    matrix()
    diff_recall()
    diff_start()
