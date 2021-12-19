#include <cmath>
#include <cstdio>
#include <ctime>
#include <set>
#include <array>
#include <fstream>

#define iterations 2000

#define a 28.7
#define b 0.22
#define c 0.7

#define d 1
#define max_index 240
#define min_index -50
#define t_limit 5000
#define base 1.05
#define recall_cost 1.0
#define forget_cost 1.0

using namespace std;

float cal_start_stability(int difficulty) {
    return 0.79 * pow(difficulty, -0.86);
}

float cal_next_stability(float s, float r) {
    return s * (1 + a * pow(d, -c) * pow(s, -b) * (1 - r));
}

int cal_stability_index(float s) {
    return (int) round(log(s) / log(base)) - min_index;
}

float cal_index_stability(int index) {
    return exp((index + min_index) * log(base));
}

int main() {

    auto stability_list = new float[max_index - min_index];
    for (int i = 0; i < max_index - min_index; i++) {
        stability_list[i] = pow(base, i + min_index);
    }

    int index_len = max_index - min_index;
    auto stress_list = new float[t_limit][max_index - min_index];
    for (int t = 0; t < t_limit; t++) {
        for (int i = 0; i < index_len - 1; i++) {
            stress_list[t][i] = t + 1;
        }
        stress_list[t][index_len - 1] = (float) (t + 1) / cal_stability_index(index_len - 1);
    }

    array<float, 85> factor_list{};
    for (int i = 0; i < 85; i++) {
        factor_list[i] = log((i + 10) / 100.0) / log(0.9);
    }

    auto used_interval_list = new int[t_limit][max_index - min_index];
    auto recall_list = new float[t_limit][max_index - min_index];
    auto next_index = new int[t_limit][max_index - min_index];

    float s0 = cal_start_stability(d);
    int s0_index = cal_stability_index(s0);
    int start_time = (int) time((time_t *) nullptr);

    for (int i = 0; i < iterations; ++i) {
        float s0_stress = stress_list[t_limit - 1][s0_index];
        for (int day = 0; day < t_limit; day++) {
            int t_remain = t_limit - 1 - day;
            for (int s_index = index_len - 1; s_index >= s0_index; s_index--) {
                float stability = stability_list[s_index];
                set<int> interval_set;
                float use_factor_list[25];
                if (stability >= 1) {
                    for (int f = 0; f < 25; f++) {
                        use_factor_list[f] = factor_list[f + 60];
                    }
                    for (float j : use_factor_list) {
                        interval_set.insert(max(1, (int) round(stability * j)));
                    }
                } else {
                    for (float j : factor_list) {
                        interval_set.insert(max(1, (int) round(stability * j)));
                    }
                }
                for (int interval : interval_set) {
                    float recall = exp(log(0.9) * interval / stability);
                    float next_s = cal_next_stability(stability, recall);
//                float next_s = cal_next_stability_simple(stability, recall);
                    int next_s_index = cal_stability_index(next_s);
                    float next_stress;
                    float exp_stress;
                    if (t_remain - interval >= 0) {
                        if (next_s_index >= index_len) {
                            next_stress = (float) (t_remain + 1) / (float) interval;
                        } else {
                            next_stress = stress_list[t_remain - interval][next_s_index];
                        }
                        exp_stress =
                                recall * (next_stress + recall_cost) +
                                (1.0 - recall) * (stress_list[t_remain - interval][s0_index] + forget_cost);
                    } else {
                        exp_stress = (float) (t_remain + 1) / (float) interval;
                    }
                    if (exp_stress < stress_list[t_remain][s_index]) {
                        stress_list[t_remain][s_index] = exp_stress;
                        used_interval_list[t_remain][s_index] = interval;
                        recall_list[t_remain][s_index] = recall;
                        next_index[t_remain][s_index] = next_s_index;
                    }
                }
            }
//            for (int s_index = s0_index; s_index <= index_len - 1; s_index++) {
//                float stability = stability_list[s_index];
//                set<int> interval_set;
//                float use_factor_list[40];
//                if (stability >= 1) {
//                    for (int f = 0; f < 40; f++) {
//                        use_factor_list[f] = factor_list[f + 50];
//                    }
//                    for (float j : use_factor_list) {
//                        interval_set.insert(max(1, (int) round(stability * j)));
//                    }
//                } else {
//                    for (float j : factor_list) {
//                        interval_set.insert(max(1, (int) round(stability * j)));
//                    }
//                }
//                for (int interval : interval_set) {
//                    float recall = exp(log(0.9) * interval / stability);
//                    float next_s = cal_next_stability(stability, recall);
////                float next_s = cal_next_stability_simple(stability, recall);
//                    int next_s_index = cal_stability_index(next_s);
//                    float next_stress;
//                    float exp_stress;
//                    if (t_remain - interval > 0) {
//                        if (next_s_index >= index_len) {
//                            next_stress = 0;
//                        } else {
//                            next_stress = stress_list[t_remain - interval][next_s_index];
//                        }
//                        exp_stress =
//                                recall * next_stress + (1.0 - recall) * stress_list[t_remain - interval][s0_index] + 1.0;
//                    } else {
//                        exp_stress = (float) t_remain / (float) interval;
//                    }
//                    if (exp_stress <= stress_list[t_remain][s_index]) {
//                        stress_list[t_remain][s_index] = exp_stress;
//                        used_interval_list[t_remain][s_index] = interval;
//                        recall_list[t_remain][s_index] = recall;
//                        next_index[t_remain][s_index] = next_s_index;
//                    }
//                }
//            }
        }


        float diff = s0_stress - stress_list[t_limit - 1][s0_index];
        if (i % 5 == 0) {
            ofstream used_interval_out("used_interval.csv");
            ofstream stress_out("stress.csv");
            ofstream recall_out("recall.csv");
            for (int j = 0; j < t_limit; j++) {
                for (int k = 0; k < index_len; k++) {
                    used_interval_out << used_interval_list[j][k] << ',';
                    stress_out << stress_list[j][k] << ',';
                    recall_out << recall_list[j][k] << ',';
                }
                used_interval_out << '\n';
                stress_out << '\n';
                recall_out << '\n';
            }

            int t_remain = t_limit - 1;
            int s_index = s0_index;
            int ivl = used_interval_list[t_remain][s_index];
            float stress = stress_list[t_remain][s_index];
            float recall = recall_list[t_remain][s_index];
            do {
                float s = cal_index_stability(s_index);
                printf("s:%10.4f\tivl:%5d\tr:%.4f\tstress:%10.4f\tt:%5d\n", s, ivl, recall, stress, t_remain);
                s_index = next_index[t_remain][s_index];
                t_remain = t_remain - ivl;
                if (t_remain < 0 || ivl <= 0) break;
                ivl = used_interval_list[t_remain][s_index];
                recall = recall_list[t_remain][s_index];
                stress = stress_list[t_remain][s_index];
            } while (t_remain > 0 && s_index < index_len);
            printf("iter %d\tdiff %f\ttime %ds\tstress %f\n", i, diff, (int) time((time_t *) nullptr) - start_time,
                   stress_list[t_limit - 1][s0_index]);
            if (diff < 0.00001 && i > 10) {
                break;
            }
        }

    }
//    FILE *f;
//    char name[40];
//    sprintf(name, "stress-ivl-%d-%d.csv", max_index, d);
//    f = fopen(name, "w");
//    fprintf(f, "当前稳定性,最小复习压力,最优可提取性,复习间隔\n");
//    for (int k = s0_index; k < index_len && k > 0; k = next_index[k])
//        fprintf(f, "%lf,%Lf,%lf,%d\n", stability_list[k], stress_list[k], recall_list[k], used_interval_list[k]);
//    fclose(f);
    return 0;
}