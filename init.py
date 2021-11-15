import matplotlib.pyplot as plt
import random

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.rcParams['figure.dpi'] = 300

start_stability = 1
a = -16
b = 0.23

random.seed(114514)
