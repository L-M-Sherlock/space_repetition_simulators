import pandas as pd
import numpy as np
from init import *
from sinc_fun import stability_inc_linear, stability_inc_log, stability_inc_exp

if __name__ == "__main__":
    random.seed(114)
    value_thresholds = [0.2, 0.3, 0.4]
    period_len = 60  # 滚动平均区间
    learn_days = 360 * 3  # 模拟时长
    deck_size = 100000  # 新卡片总量
    card_per_day_limit = 400
    learn_limit = 400
    review_limit = 400

    for i, v in enumerate(value_thresholds):
        value = value_thresholds[i]

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
            with np.errstate(divide='ignore'):
                df_card["V"] = df_card["R"] * (
                        np.exp(np.log(0.9) / (df_card["S"] * stability_inc_exp(df_card["S"], df_card["R"]))) - np.exp(
                    np.log(0.9)) - 1) + np.exp(np.log(0.9))
            record_per_day[day] = df_card[df_card["S"] >= start_stability]["R"].sum()
            # value = 0.9 / (df_card[df_card["R"] > 0]["R"].count() + 1) * review_limit
            review = df_card[(df_card["S"] >= start_stability) & (df_card["V"] >= value)].sort_values(by=['V'],
                                                                                                      ascending=[
                                                                                                          False]).index[
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

            learn = df_card[df_card["S"] < start_stability].index[
                    :min(learn_limit, card_per_day_limit - real_review_num)]
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

        recall = df_card[df_card["S"] >= start_stability]["R"].sum()
        total_learned = int(sum(new_card_per_day))
        total_reviewed = int(sum(workload_per_day)) - total_learned

        plt.figure(1)
        plt.plot(record_per_day, label=f'threshold={value:.2f}|E(M)={recall:.2f}')
        # plt.show()

        plt.figure(2)
        plt.plot(new_card_per_day_average_per_period, label=f'threshold={value:.2f}|learned={total_learned}')
        plt.ylim((0, card_per_day_limit + 10))
        print(df_card[df_card["R"] > 0]["R"].mean())
    # plt.title(f"{learn_days}天-遗忘比例{1 - expected_recall:.2f}-总学习量{total_learned}-记忆保留总量{int(recall)}")

    # plt.show()
    # plt.plot(workload_per_day_average_per_period-new_card_per_day_average_per_period,
    #          label=f'遗忘指数{1 - expected_recall:.2f}/保留量{int(recall)}/总复习量{total_reviewed}')
    # plt.title(f"总学习量{total_learned}")
    # plt.xlabel("时间/天")
    # plt.ylabel(f"每日复习卡片数量({period_len}天平均)")
    # plt.show()
    plt.figure(1)
    plt.title(f"Greedy-每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel("记住的单词数量期望E(W)")
    plt.legend()
    plt.grid(True)
    plt.figure(2)
    plt.title(f"Greedy-每日学习上限:{card_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel(f"每日新学数量({period_len}天平均)")
    plt.legend()
    plt.grid(True)
    plt.show()
