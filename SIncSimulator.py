import numpy as np
import random
from collections import deque
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

start_stability = 1
a = -16
b = 0.23


class Item:

    def __init__(self):
        self.SInc = 0
        self.R = 0
        self.feedback = 0
        self.used_ivl = 0
        self.S = start_stability
        self.last_review = -1
        self.ivl_history = []
        self.fb_history = []
        self.lapse = 0
        self.difficulty = round(random.triangular(), 1)

    def review(self, date):
        self.__calculate_r(date)
        self.feedback = 1 if random.random() < self.R else 0
        self.fb_history.append(self.feedback)
        self.used_ivl = 0 if self.last_review < 0 else date - self.last_review
        self.ivl_history.append(self.used_ivl)
        if self.feedback == 0:
            self.lapse += 1
            self.S = start_stability
        else:
            self.__calculate_sinc()
            self.__update_s()
        self.last_review = date

    def __calculate_r(self, date):
        if self.last_review >= 0:
            self.R = np.exp(np.log(0.9) * (date - self.last_review) / self.S)
        else:
            self.R = 0.9

    def __calculate_sinc(self):
        # self.SInc = - a * np.power(self.S, -b) * (np.exp(1 - self.R) - 1) * 1.1 + 1
        self.SInc = (5 * (1 - self.difficulty) + 1) * (a * np.power(self.S, -b) * np.log(self.R)) + 1

    def __update_s(self):
        self.S *= self.SInc

    def interval_threshold(self, threshold):
        return int(round(max(1, np.log(threshold) / np.log(0.9) * self.S), 0))

    def interval_max_esinc(self):
        def esinc(x):
            return - x * (a * np.power(self.S, -b) * np.log(x) + 1)

        def esinc2(x):
            return - x * (- 1.1 * a * np.power(self.S, -b) * (np.exp(1 - x) - 1) + 1)

        r = optimize.fminbound(esinc2, 0, 1)
        return int(round(max(1, np.log(r) / np.log(0.9) * self.S), 0))


if __name__ == '__main__':

    learn_days = 600  # 模拟时长
    deck_size = 1000000  # 新卡片总量
    items_all = [Item() for _ in range(0, deck_size)]
    items = deque(items_all)
    item_per_day = [[] for _ in range(0, learn_days)]
    card_per_day_limit = 100000  # 每日学习总上限
    new_card_per_day_limit = 1000
    card_per_day = [{'forget': 0, 'recall': 0, 'new': 0} for i in range(0, learn_days)]
    end_day = 0
    forget_index = 0.1  # 遗忘比率
    period_len = 200  # 滚动平均区间
    new_card_per_day = np.array([0.0] * learn_days)
    new_card_per_day_average_per_period = np.array([0.0] * learn_days)
    workload_per_day = np.array([0.0] * learn_days)
    workload_per_day_average_per_period = np.array([0.0] * learn_days)
    revlog = np.empty(shape=[0, 7], dtype=object)

    for day in range(0, learn_days):
        card_per_day[day]['new'] = min(max(0, card_per_day_limit - len(item_per_day[day])),
                                       new_card_per_day_limit) if len(items) > 0 else 0
        i = 0
        while len(item_per_day[day]) < card_per_day_limit and len(items) > 0 and i < new_card_per_day_limit:
            item_per_day[day].append(items.pop())
            i += 1

        new_card_per_day[day] = card_per_day[day]['new']

        if day >= period_len:
            new_card_per_day_average_per_period[day] = np.true_divide(new_card_per_day[day - period_len:day].sum(),
                                                                      period_len)
        else:
            new_card_per_day_average_per_period[day] = np.true_divide(new_card_per_day[:day + 1].sum(), day + 1)

        if len(items) > 0:
            end_day = day
        today_revlog = np.empty(shape=[len(item_per_day[day]), 7], dtype=object)
        for idx, item in enumerate(item_per_day[day]):
            ivl_history = str(item.ivl_history)[1:-1]
            fb_history = str(item.fb_history)[1:-1]
            old_S = round(item.S, 1)
            item.review(day)
            today_revlog[idx] = [ivl_history, fb_history, item.used_ivl, item.feedback, round(item.R, 3), old_S,
                                 item.difficulty]
            forget_index = random.triangular(0.05, 0.2, 0.1)
            ivl = int(round(item.interval_threshold(1 - forget_index)))
            # ivl = int(round(ivl) * 5 * round(random.uniform(0.15, 1.04), 1))
            # ivl = item.interval_max_esinc()
            delay = 0
            while day + ivl + delay < learn_days and len(item_per_day[day + ivl + delay]) >= card_per_day_limit:
                delay += 1
            if day + ivl + delay >= learn_days:
                continue
            item_per_day[day + ivl + delay].append(item)
            if item.feedback == 0:
                card_per_day[day + ivl + delay]['forget'] += 1
            else:
                card_per_day[day + ivl + delay]['recall'] += 1
        revlog = np.concatenate((revlog, today_revlog))
    # recall = 0
    # for item in items_all:
    #     if item.last_review > 0:
    #         item.review(learn_days + 1)
    #         recall += item.R
    print("数据条数", len(revlog))
    result = pd.DataFrame(revlog, columns=["ivl_history", "fb_history", "used_ivl", "feedback", "R", "S", "D"])
    result.to_csv(f"revlog{int(time.time())}.tsv", sep="\t", index=False)
    total_learned = int(sum(new_card_per_day))
    print("learn all item day:", end_day + 1)
    plt.plot(new_card_per_day_average_per_period)
    plt.xlabel("时间/天")
    plt.ylabel(f"每日新卡片数量{period_len}天平均)")
    # plt.title(f"{learn_days}天/遗忘比例{forget_index:.2f}/总学习量{total_learned}/记忆保留总量{int(recall)}")
    plt.grid(True)
    plt.show()
