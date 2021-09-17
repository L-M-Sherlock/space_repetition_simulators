from init import *
from sinc_fun import stability_inc_linear, stability_inc_log, stability_inc_exp
import numpy as np


def get_r(RI, ISI):
    start_stability = 5
    r = np.exp(np.log(0.9) * ISI / start_stability)
    return r * np.exp(np.log(0.9) * RI / (start_stability * stability_inc_linear(start_stability, r))) + (1 - r) * np.exp(
        np.log(0.9) * RI / start_stability)


if __name__ == "__main__":
    RIs = [7, 35, 70, 350]
    ISIs = range(1, 105)
    for RI in RIs:
        Rs_in_RI = []
        for ISI in ISIs:
            Rs_in_RI.append(get_r(RI, ISI))
        plt.plot(ISIs, Rs_in_RI, '-', label=f'{RI} Day retention')
    plt.ylim(-0.05, 1)
    plt.xlim(0, 105)
    plt.ylabel("% Recall")
    plt.xlabel("Spacing (days)")
    plt.legend()
    plt.show()
