#include <iostream>
#include <cmath>
#include <numeric>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include <set>

#define iterations 200000
#define inf 10001.0

#define a 28.7
#define b 0.22
#define c 0.7

#define d 1
#define max_index (200 - 10 * d)
#define min_index -50
#define recall_cost 1
#define forget_cost 1
#define base 1.05

using namespace std;

double cal_start_stability(int difficulty) {
    return 0.79 * pow(difficulty, -0.86);
}

double cal_next_stability(double s, double r) {
    return s * (1 + a * pow(d, -c) * pow(s, -b) * (1 - r));
}

int cal_stability_index(double s) {
    return round(log(s) / log(base)) - min_index;
}

int main() {

    double stability_list[max_index - min_index] = {0.0};
    for (int i = 0; i < max_index - min_index; i++) {
        stability_list[i] = pow(base, i + min_index);
    }

    int index_len = max_index - min_index;

    long double stress_list[max_index - min_index] = {0.0};
    for (long double &stress : stress_list) {
        stress = inf;
    }
    stress_list[index_len - 1] = 0.0;

    double factor_list[90];
    for (int i = 0; i < 90; i++) {
        factor_list[i] = log((i + 10) / 100.0) / log(0.9);
    }

    int used_interval_list[max_index - min_index] = {0};
    double recall_list[max_index - min_index] = {0.0};
    int next_index[max_index - min_index] = {0};

    double s0 = cal_start_stability(d);
    int s0_index = cal_stability_index(s0);
    int start_time = (int) time((time_t *) NULL);

    for (int i = 0; i < iterations; ++i) {
        long double s0_stress = stress_list[s0_index];
        for (int s_index = index_len - 1; s_index >= s0_index; s_index--) {
            double stability = stability_list[s_index];
            set<int> interval_set;
            for (int j = 0; j < 90; j++) {
                interval_set.insert(max(1, (int) round(stability * factor_list[j])));
            }
            for (set<int>::iterator ivl = interval_set.begin(); ivl != interval_set.end(); ivl++) {
                int interval = *ivl;
                double recall = exp(log(0.9) * interval / stability);
                double next_s = cal_next_stability(stability, recall);
                int next_s_index = cal_stability_index(next_s);
                long double next_stress;
                if (next_s_index >= index_len) {
                    next_stress = 0;
                } else {
                    next_stress = stress_list[next_s_index];
                }
                long double exp_stress = recall * next_stress + (1 - recall) * stress_list[s0_index] + 1;
                if (exp_stress < stress_list[s_index]) {
                    stress_list[s_index] = exp_stress;
                    used_interval_list[s_index] = interval;
                    recall_list[s_index] = recall;
                    next_index[s_index] = next_s_index;
                }
            }
        }
        for (int s_index = s0_index; s_index <= index_len - 1; s_index++) {
            double stability = stability_list[s_index];
            set<int> interval_set;
            for (int j = 0; j < 90; j++) {
                interval_set.insert(max(1, (int) round(stability * factor_list[j])));
            }
            for (set<int>::iterator ivl = interval_set.begin(); ivl != interval_set.end(); ivl++) {
                int interval = *ivl;
                double recall = exp(log(0.9) * interval / stability);
                double next_s = cal_next_stability(stability, recall);
                int next_s_index = cal_stability_index(next_s);
                long double next_stress;
                if (next_s_index >= index_len) {
                    next_stress = 0;
                } else {
                    next_stress = stress_list[next_s_index];
                }
                long double exp_stress = recall * next_stress + (1 - recall) * stress_list[s0_index] + 1;
                if (exp_stress < stress_list[s_index]) {
                    stress_list[s_index] = exp_stress;
                    used_interval_list[s_index] = interval;
                    recall_list[s_index] = recall;
                    next_index[s_index] = next_s_index;
                }
            }
        }
        long double diff = s0_stress - stress_list[s0_index];
        if (i % 100 == 0)
            printf("iter %d, diff %Lf, time %ds, stress %Lf\n", i, diff, (int) time((time_t *) NULL) - start_time,
                   stress_list[s0_index]);
        if (diff < 0.00001 && i > 100) {
            break;
        }
    }
    FILE *f;
    char name[40];
    sprintf(name, "stress-ivl-%d-%d.csv", max_index, d);
    f = fopen(name, "w");
    fprintf(f, "当前稳定性,最小复习压力,最优可提取性,复习间隔\n");
    for (int k = s0_index; k < index_len && k > 0; k++)
        fprintf(f, "%lf,%Lf,%lf,%d\n", stability_list[k], stress_list[k], recall_list[k], used_interval_list[k]);
    fclose(f);
    return 0;
}