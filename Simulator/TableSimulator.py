import matplotlib.pyplot as plt

from init import *
from sinc_fun import stability_inc_linear, stability_inc_log, stability_inc_exp
import pandas as pd
import numpy as np
import random

if __name__ == "__main__":
    expected_recalls = [0.8]
    period_len = 60  # 滚动平均区间
    learn_days = 360  # 模拟时长
    deck_size = 100000  # 新卡片总量
    card_per_day_limit = 1000
    learn_limit = 50
    review_limit = 1000
    easy_first = False
    for i, v in enumerate(expected_recalls):
        expected_recall = expected_recalls[i]
        random.seed(114514)
        new_card_per_day = np.array([0.0] * learn_days)
        new_card_per_day_average_per_period = np.array([0.0] * learn_days)
        workload_per_day = np.array([0.0] * learn_days)
        workload_per_day_average_per_period = np.array([0.0] * learn_days)
        record_per_day = np.array([0.0] * learn_days)
        delta_record_per_day = np.array([0.0] * learn_days)

        feature_list = ["ivl_history", "fb_history", "review_date", "R", "S", "D", "ivl"]
        dtypes = np.dtype([
            ('ivl_history', str),
            ('fb_history', str),
            ('review_date', int),
            ('R', float),
            ('S', float),
            ('D', float),
            ('ivl', int),
        ])

        df_card = pd.DataFrame(np.full(deck_size, np.nan, dtype=dtypes), index=range(deck_size), columns=feature_list)

        for day in range(learn_days):
            df_card["ivl"] = day - df_card["review_date"]
            df_card["R"] = np.exp(np.log(0.9) * df_card["ivl"] / df_card["S"])
            if easy_first:
                review = df_card[df_card["R"] <= expected_recall].sort_values(by=['R', 'S'],
                                                                              ascending=[False, False]).index[
                         :review_limit]
            else:
                review = df_card[df_card["R"] <= expected_recall].sort_values(by=['R', 'S'],
                                                                              ascending=[True, True]).index[
                         :review_limit]
            real_review_num = len(review)
            for idx in review:
                df_card.iat[idx, 2] = day
                df_card.iat[idx, 0] += str(df_card.iat[idx, 6]) + ','
                if random.random() < df_card.iat[idx, 3]:
                    df_card.iat[idx, 1] += '1'
                    df_card.iat[idx, 4] *= stability_inc_exp(df_card.iat[idx, 4], df_card.iat[idx, 3])
                else:
                    df_card.iat[idx, 1] += '0'
                    df_card.iat[idx, 4] = start_stability

            learn = df_card[df_card["S"].isna()].index[:min(learn_limit, card_per_day_limit - real_review_num)]
            for idx in learn:
                df_card.iat[idx, 2] = day
                df_card.iat[idx, 4] = start_stability

            new_card_per_day[day] = len(learn)
            workload_per_day[day] = new_card_per_day[day] + real_review_num

            if day >= period_len:
                new_card_per_day_average_per_period[day] = np.true_divide(new_card_per_day[day - period_len:day].sum(),
                                                                          period_len)
                workload_per_day_average_per_period[day] = np.true_divide(workload_per_day[day - period_len:day].sum(),
                                                                          period_len)
            else:
                new_card_per_day_average_per_period[day] = np.true_divide(new_card_per_day[:day + 1].sum(), day + 1)
                workload_per_day_average_per_period[day] = np.true_divide(workload_per_day[:day + 1].sum(), day + 1)


            df_card["ivl"] = day - df_card["review_date"]
            df_card["R"] = np.exp(np.log(0.9) * df_card["ivl"] / df_card["S"])
            record_per_day[day] = df_card[df_card["S"] >= start_stability]["R"].sum()
            delta_record_per_day[day] = record_per_day[day] - record_per_day[day - 1]

        recall = df_card[df_card["S"] >= start_stability]["R"].sum()
        total_learned = int(sum(new_card_per_day))
        total_reviewed = int(sum(workload_per_day)) - total_learned

        plt.figure(1)
        plt.plot(record_per_day, label=f'E(M)={recall:.2f}')

        plt.figure(2)
        plt.plot(delta_record_per_day, label=f'dE(M)={recall:.2f}')

        plt.figure(3)
        plt.plot(new_card_per_day_average_per_period, label=f'learned={total_learned}')
        plt.ylim((0, card_per_day_limit + 10))

        plt.figure(4)
        plt.plot(workload_per_day_average_per_period, label=f'reviewed={total_reviewed}')

        print("R_min", df_card["R"].min())
        print("R_mean", df_card[df_card["R"] > 0]["R"].mean())

    plt.figure(1)
    plt.title(f"每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel("记住的单词数量期望E(W)")
    plt.legend()
    plt.grid(True)
    plt.figure(2)
    plt.title(f"每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel("记住的单词数量期望E(W)增量")
    plt.legend()
    plt.grid(True)
    plt.figure(3)
    plt.title(f"每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel(f"每日新学数量({period_len}天平均)")
    plt.legend()
    plt.grid(True)
    plt.figure(4)
    plt.title(f"每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel(f"每日学习数量({period_len}天平均)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.hist(x=df_card['R'], range=(0, 1), bins=20)
    plt.ylabel("count")
    plt.xlabel('R')
    plt.title(f"每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.show()
