import numpy as np
import time
from scipy import optimize
from init import *

learn_day_limit = 365 * 10


def expected_learn(acc_prob, stability, day, record, fi):
    if day >= learn_day_limit or acc_prob <= np.power(0.1, 5):
        return

    record[day] += acc_prob
    if fi == -1:
        def min_stress(x):
            return x * 1 / (np.log(stability * (a * np.power(stability, -b) * np.log(x) + 1))) + (
                        1 - x) * 1 / start_stability + np.log(0.9)/np.log(x)

        R = optimize.fminbound(min_stress, 0.01, 0.99)
    else:
        R = 1 - fi
    if stability > 0:
        s_recall = stability * (a * np.power(stability, -b) * (R - 1) + 1)
    else:
        s_recall = start_stability
    ivl_recall = int(round(max(1, np.log(R) / np.log(0.9) * s_recall), 0))
    s_forget = start_stability
    ivl_forget = int(round(max(1, np.log(R) / np.log(0.9) * s_forget), 0))
    expected_learn(acc_prob * R, s_recall, day + ivl_recall, record, fi)
    expected_learn(acc_prob * (1 - R), s_forget, day + ivl_forget, record, fi)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fis = [0.2, 0.4, 0.6, -1]
    for fi in fis:
        record = np.zeros(learn_day_limit)
        print("start")
        start = time.time()
        expected_learn(1, 0, 0, record, fi)
        print(f"end with {time.time() - start}s")
        # plt.plot(record)
        # plt.show()
        for idx in range(1, len(record)):
            record[idx] += record[idx - 1]
        plt.plot(record, label=f'遗忘指数{fi:.2f}/保留率{-fi / np.log(1 - fi)}')
    plt.grid(True)
    plt.legend()
    plt.xlabel("天数")
    plt.ylabel("期望复习次数")
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
