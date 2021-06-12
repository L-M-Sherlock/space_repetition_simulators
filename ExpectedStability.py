import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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
        es = np.power(r, k + 1) * stability(k+1, r) + (1 - r) * es
    return es


if __name__ == "__main__":
    random.seed(114)
    estability_matrix = np.empty(shape=[10, 9])
    for i in range(1, 11):
        for j in range(1, 10):
            reps = i
            r = j / 10
            estability_matrix[i - 1][j - 1] = estability(reps, r)
    print(estability_matrix)
    recalls = [0.9, 0.8, 0.6, 0.4, 0.3]
    for recall in recalls:
        repeats = [list() for _ in range(10000)]
        for _ in range(10000):
            day = 0
            s = 1
            reps = 0
            while day < 10000:
                if random.random() < recall:
                    s *= a * np.power(s, -b) * np.log(recall) + 1
                else:
                    s = np.log(recall) / np.log(0.9)
                repeats[day].append(reps)
                reps += 1
                day += round(s)
        avg_repeats = []
        for repeat in repeats:
            if len(repeat) > 0:
                avg_repeats.append(sum(repeat)/len(repeat))
            else:
                avg_repeats.append(0)
        plt.plot(savgol_filter(avg_repeats, 999, 1), "-", label=f'保留率{recall}')

    plt.grid(True)
    plt.xlabel("时间/天")
    plt.legend()
    plt.show()