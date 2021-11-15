from init import *
from sinc_fun import stability_inc_slow, stability_inc_linear
import numpy as np

if __name__ == "__main__":
    start_stability = 1

    review_groups = [
        # [4, 12, 20, 28],
        [2, 6, 10, 14],
        [1, 3, 5, 7, 15],
        [3, 9, 20]
    ]
    colors = ['green', 'blue', 'red', 'orange']

    for id0, review_points in enumerate(review_groups):
        r = []
        s = [start_stability]
        t = np.arange(200)
        y = np.exp(np.log(0.9) * t[:review_points[0]] / start_stability)
        plt.plot(t[:review_points[0]], y, color=colors[id0], alpha=0.5)
        for id1, ivl in enumerate(review_points):
            r.append(np.exp(np.log(0.9) * ivl / s[-1]))
            s.append(s[-1] * stability_inc_linear(s[-1], r[-1]))
            start_point = sum(review_points[:id1 + 1]) - 1
            plot_ivl = review_points[id1 + 1] + 1 if id1 + 1 < len(review_points) else 30
            # s.append(s[-1] * stability_inc_linear(s[-1], r[-1]))
            # y = np.exp(np.log(0.9) * t / s[-1]) * r[-1] + np.exp(np.log(0.9) * t / s[0]) * (1 - r[-1])

            plt.plot([start_point] * 2, [y[-1], 1], color=colors[id0], alpha=0.5)
            y = np.exp(np.log(0.9) * t[:plot_ivl] / s[-1])
            # s[-1] = s[-1] * r[-1] + s[0] * (1 - r[-1])
            plt.plot(t[:plot_ivl] + start_point, y, color=colors[id0], alpha=0.5)
    plt.show()
