import pandas as pd
import numpy as np
from init import *
from sinc_fun import stability_inc_linear, stability_inc_log, stability_inc_exp
from tqdm import tqdm

remember = 3
forget = 15

if __name__ == "__main__":
    random.seed(114514)
    values = [0]
    period_len = 15  # 滚动平均区间
    learn_days = 30000  # 模拟时长
    deck_size = 100000  # 新卡片总量
    card_per_day_limit = 100
    learn_limit = 100
    review_limit = 100
    time_per_day = 300  # 每天背单词的时间，单位秒
    # review_table = np.loadtxt(open("30000-exp-stress-time.csv", "rb"), delimiter=",", skiprows=1)

    for i, v in enumerate(values):
        value = values[i]
        total_time = 0

        new_card_per_day = np.array([0.0] * learn_days)
        new_card_per_day_average_per_period = np.array([0.0] * learn_days)
        workload_per_day = np.array([0.0] * learn_days)
        workload_per_day_average_per_period = np.array([0.0] * learn_days)
        record_per_day = np.array([0.0] * learn_days)
        delta_record_per_day = np.array([0.0] * learn_days)

        feature_list = ["ivl_history", "fb_history", "review_date", "R", "S", "V", "ivl"]
        dtypes = np.dtype([
            ('ivl_history', str),
            ('fb_history', str),
            ('review_date', int),
            ('R', float),
            ('S', float),
            ('V', float),
            ('ivl', int),
            ('diff', float)
        ])

        df_card = pd.DataFrame(np.full(deck_size, np.nan, dtype=dtypes), index=range(deck_size), columns=feature_list)

        for day in tqdm(range(learn_days)):
            day_time = 0
            reviewed = 0
            learned = 0
            df_card["ivl"] = day - df_card["review_date"]
            df_card["R"] = np.exp(np.log(0.9) * df_card["ivl"] / df_card["S"])
            T = learn_days - day
            df_card["diff"] = df_card["R"] - df_card["V"]
            review = df_card[(df_card["diff"] <= value)].sort_values(by=['diff'],
                                                                     ascending=[True]).index[:review_limit]
            for idx in review:
                if day_time > time_per_day:
                    total_time += day_time
                    break
                reviewed += 1
                df_card.iat[idx, 2] = day
                df_card.iat[idx, 0] += str(df_card.iat[idx, 6]) + ','
                if random.random() < df_card.iat[idx, 3]:
                    df_card.iat[idx, 1] += '1'
                    df_card.iat[idx, 4] *= stability_inc_exp(df_card.iat[idx, 4], df_card.iat[idx, 3])
                    # df_card.iat[idx, 5] = review_table[round(df_card.iat[idx, 4]) - 1][1]
                    df_card.iat[idx, 5] = 0.008 * np.log(df_card.iat[idx, 4]) + 0.8
                    day_time += remember
                else:
                    df_card.iat[idx, 1] += '0'
                    df_card.iat[idx, 4] = start_stability
                    # df_card.iat[idx, 5] = review_table[round(df_card.iat[idx, 4]) - 1][1]
                    df_card.iat[idx, 5] = 0.008 * np.log(df_card.iat[idx, 4]) + 0.8

                    day_time += forget

            # learn = df_card[df_card["S"] < start_stability].index[
            #         :min(learn_limit, card_per_day_limit - real_review_num)]
            learn = df_card[df_card["S"].isna()].index[:min(learn_limit, card_per_day_limit - reviewed)]
            for idx in learn:
                if day_time > time_per_day:
                    total_time += day_time
                    break
                learned += 1
                df_card.iat[idx, 2] = day
                df_card.iat[idx, 4] = start_stability
                # df_card.iat[idx, 5] = review_table[start_stability - 1][1]
                df_card.iat[idx, 5] = 0.008 * np.log(df_card.iat[idx, 4]) + 0.8

            new_card_per_day[day] = learned
            workload_per_day[day] = learned + reviewed

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
            total_time += day_time

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
        print("E(M) per second", recall / total_time)

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
