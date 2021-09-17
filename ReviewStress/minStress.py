import pandas as pd
import numpy as np
from init import *
from sinc_fun import stability_inc_linear, stability_inc_log, stability_inc_exp


def review_value(s, r):
    return - (s * r * (stability_inc_log(s, r) - 1) + start_stability * (1 - r)) / np.log(0.9)


if __name__ == "__main__":
    values = [0]
    period_len = 60  # 滚动平均区间
    learn_days = 360 * 10  # 模拟时长
    deck_size = 500000  # 新卡片总量
    card_per_day_limit = 200
    learn_limit = 200
    review_limit = 200
    review_table = np.loadtxt(open("30000-stress.csv", "rb"), delimiter=",", skiprows=0)

    for i, v in enumerate(values):
        value = values[i]
        random.seed(114)

        new_card_per_day = np.array([0.0] * learn_days)
        new_card_per_day_average_per_period = np.array([0.0] * learn_days)
        workload_per_day = np.array([0.0] * learn_days)
        workload_per_day_average_per_period = np.array([0.0] * learn_days)
        record_per_day = np.array([0.0] * learn_days)

        feature_list = ["ivl_history", "fb_history", "review_date", "R", "S", "V", "ivl"]
        dtypes = np.dtype([
            ('ivl_history', str),
            ('fb_history', str),
            ('review_date', int),
            ('R', float),
            ('S', int),
            ('V', float),
            ('ivl', int),
            ('diff', float)
        ])

        df_card = pd.DataFrame(np.full(deck_size, np.nan, dtype=dtypes), index=range(deck_size), columns=feature_list)

        for day in range(learn_days):
            df_card["ivl"] = day - df_card["review_date"]
            df_card["R"] = np.exp(np.log(0.9) * df_card["ivl"] / df_card["S"])
            # df_card["V"] = - ((df_card["S"] * df_card["R"] * (
            #             stability_inc_log(df_card["S"], df_card["R"]) - 1)) + start_stability * (
            #                               1 - df_card["R"])) / np.log(0.9)
            T = learn_days - day
            # df_card["V"] = df_card["S"] * (df_card["R"] - np.power(0.9, T / df_card["S"]))
            record_per_day[day] = df_card[df_card["S"] >= start_stability]["R"].sum()
            df_card["diff"] = df_card["R"] - df_card["V"]
            review = df_card[(df_card["S"] >= start_stability) & (df_card["diff"] <= value)].sort_values(by=['diff'],
                                                                                                         ascending=[
                                                                                                             True]).index[
                     :review_limit]
            real_review_num = len(review)
            for idx in review:
                df_card.iat[idx, 2] = day
                df_card.iat[idx, 0] += str(df_card.iat[idx, 6]) + ','
                if random.random() < df_card.iat[idx, 3]:
                    df_card.iat[idx, 1] += '1'
                    df_card.iat[idx, 4] *= stability_inc_log(df_card.iat[idx, 4], df_card.iat[idx, 3])
                    df_card.iat[idx, 5] = review_table[df_card.iat[idx, 4] - 1][1]
                else:
                    df_card.iat[idx, 1] += '0'
                    df_card.iat[idx, 4] = start_stability
                    df_card.iat[idx, 5] = review_table[df_card.iat[idx, 4] - 1][1]

            learn = df_card[df_card["S"] < start_stability].index[
                    :min(learn_limit, card_per_day_limit - real_review_num)]
            for idx in learn:
                df_card.iat[idx, 2] = day
                df_card.iat[idx, 4] = start_stability
                df_card.iat[idx, 5] = review_table[start_stability - 1][1]

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

        recall = df_card[df_card["S"] >= start_stability]["R"].sum()
        total_learned = int(sum(new_card_per_day))
        total_reviewed = int(sum(workload_per_day)) - total_learned

        plt.figure(1)
        plt.plot(record_per_day, label=f'E(M)={recall:.2f}')
        # plt.show()

        plt.figure(2)
        plt.plot(new_card_per_day_average_per_period, label=f'learned={total_learned}')
        plt.ylim((0, card_per_day_limit + 10))
        print(df_card["R"].min())
        # plt.title(f"{learn_days}天-遗忘比例{1 - expected_recall:.2f}-总学习量{total_learned}-记忆保留总量{int(recall)}")

        # plt.show()
        # plt.plot(workload_per_day_average_per_period-new_card_per_day_average_per_period,
        #          label=f'遗忘指数{1 - expected_recall:.2f}/保留量{int(recall)}/总复习量{total_reviewed}')
        # plt.title(f"总学习量{total_learned}")
        # plt.xlabel("时间/天")
        # plt.ylabel(f"每日复习卡片数量({period_len}天平均)")
        # plt.show()
    plt.figure(1)
    plt.title(f"每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel("记住的单词数量期望E(W)")
    plt.legend()
    plt.grid(True)
    plt.figure(2)
    plt.title(f"每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel(f"每日新学数量({period_len}天平均)")
    plt.legend()
    plt.grid(True)
    plt.show()
