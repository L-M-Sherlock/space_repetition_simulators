import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

interval_rate = 2.5  # 间隔系数
fix_retention = 0.9  # 间隔系数为 2.5 时的保留率基准
request_retention = 0.9  # 用户预期的保留率
start_stability = 4  # 初始稳定性
first_interval = 4  # 初始间隔
forget_stability = 4  # 遗忘稳定性
forget_interval = 4  # 遗忘间隔
fix_first_interval = True  # 是否固定第一个间隔
forget_index = 1 - request_retention  # 遗忘比率
interval_modifier = np.log(request_retention) / np.log(fix_retention)  # 间隔修饰
deck_size = 1000  # 新卡片总量
card_per_day_limit = 999  # 每日学习总上限
new_card_limit = 50  # 每日新卡片上限
learn_days = 30  # 模拟时长
interval_limit = learn_days  # 最大间隔
period_len = 5  # 滚动平均区间
avg_reps_forget = 1
avg_reps_new = 1

end_day = 0
card_per_day = [{'forget': 0, 'recall': 0, 'new': 0} for i in range(0, learn_days)]
new_card_per_day_average_per_period = np.array([0.0] * learn_days)
workload_per_day = np.array([0.0] * learn_days)
workload_per_day_average_per_period = np.array([0.0] * learn_days)
learn_rate_per_day_average_per_period = np.array([0.0] * learn_days)
learned_per_day = np.array([0.0] * learn_days)
new_card_per_day = np.array([0.0] * learn_days)

for day in range(0, learn_days):
    cannot_delay = card_per_day[day]['forget'] + card_per_day[day]['recall']
    if cannot_delay + new_card_limit < card_per_day_limit:
        card_per_day[day]['new'] = new_card_limit
    else:
        card_per_day[day]['new'] = max(card_per_day_limit - cannot_delay, 0)
    new_card_per_day[day] = card_per_day[day]['new']
    learned_per_day[day] = learned_per_day[day - 1] + card_per_day[day]['new']
    workload_per_day[day] = card_per_day[day]['forget'] * avg_reps_forget + card_per_day[day]['recall'] + \
                            card_per_day[day]['new'] * avg_reps_new
    if day > period_len:
        workload_per_day_average_per_period[day] = np.true_divide(workload_per_day[day - period_len:day + 1].sum(),
                                                                  period_len)
        new_card_per_day_average_per_period[day] = np.true_divide(new_card_per_day[day - period_len:day + 1].sum(),
                                                                  period_len)
    else:
        workload_per_day_average_per_period[day] = np.true_divide(workload_per_day[:day + 1].sum(), day + 1)
        new_card_per_day_average_per_period[day] = np.true_divide(new_card_per_day[:day + 1].sum(), day + 1)
    i = 0
    for card in range(0, card_per_day[day]['new'] + card_per_day[day]['forget']):
        i += 1
        next_interval = first_interval if fix_first_interval else max(int(round(start_stability * interval_modifier)),
                                                                      1)
        if i > card_per_day[day]['new']:
            next_interval = start_stability
        next_due = day + next_interval
        while next_due < learn_days and next_interval < interval_limit:
            true_forget_index = 1 - np.exp(
                np.log(0.9) * next_interval / start_stability) if next_interval == first_interval else forget_index
            forget_flag = 1 if random.random() < true_forget_index else 0
            if forget_flag:
                card_per_day[next_due]['forget'] += 1
                break
            else:
                card_per_day[next_due]['recall'] += 1
            next_interval *= (interval_rate * interval_modifier)
            next_due += int(round(next_interval * random.uniform(0.9, 1.1)))
    if learned_per_day[day] >= deck_size:  # 学完所有卡片
        new_card_limit = 0
        if end_day == 0:
            end_day = day

statistics = (learn_days, forget_index, interval_modifier)
learned_per_day = np.array(learned_per_day)
plt.plot(learned_per_day)
plt.xlabel("时间/天")
plt.ylabel("学习卡片总量")
plt.title("{}天/间隔系数2.5/遗忘比例{:.2f}/间隔修饰{:.2f}倍".format(*statistics))
plt.grid(True)
plt.show()
plt.plot(new_card_per_day_average_per_period[period_len:])
plt.xlabel("时间/天")
plt.ylabel(f"每日新卡片数量({period_len}天平均)")
plt.title("{}天/间隔系数2.5/遗忘比例{:.2f}/间隔修饰{:.2f}倍".format(*statistics))
plt.grid(True)
plt.show()
plt.plot(workload_per_day_average_per_period[period_len:])
plt.xlabel("时间/天")
plt.ylabel(f"每日总点击次数({period_len}天平均)")
plt.title("{}天/间隔系数2.5/遗忘比例{:.2f}/间隔修饰{:.2f}倍".format(*statistics))
plt.grid(True)
plt.show()
learn_rate_per_day_average_per_period = np.true_divide(new_card_per_day_average_per_period,
                                                       workload_per_day_average_per_period)
plt.plot(learn_rate_per_day_average_per_period[period_len:])
plt.xlabel("时间/天")
plt.ylabel(f"学习占比({period_len}天平均)")
plt.title(f"{learn_days}天/间隔系数2.5/遗忘比例{forget_index:.2f}/间隔修饰{interval_modifier:.2f}倍")
plt.grid(True)
plt.show()
workload_total = workload_per_day.sum()
new_total = new_card_per_day.sum()
print('total repetitions:',
      workload_total,
      '\naverage of repetition for all card per day:',
      workload_total / learn_days,
      '\nvariance:',
      np.std(workload_per_day[period_len:]))
print('total learned:',
      new_total,
      '\naverage of repetition for new card per day:',
      new_total / learn_days,
      '\nvariance:',
      np.std(new_card_per_day[period_len:]))
print('learning rate:', new_total / workload_total)
print('learned all new card in', end_day)
print('true retention', -forget_index / np.log(1 - forget_index))
print('simulation end')
