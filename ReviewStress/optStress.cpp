#include <iostream>
#include <cmath>
#include <numeric>

#define max_s 30000
#define iterations 1000
#define inf 10001
#define start_stability 1
#define a -16
#define b 0.23

using namespace std;

int main() {
    double stresses[max_s] = {0};
    double review_r[max_s] = {0};
    for (double & stress : stresses) {
        stress = inf;
    }
    stresses[max_s-1] = 0;
    double sum = accumulate(stresses, stresses + max_s, 0.0);
    for (int i = 0; i < iterations; ++i) {
        for (int s_next = max_s; s_next > 0; s_next--) {
            double r = exp(pow(start_stability, b) * ((float)s_next / (float)start_stability - 1) / a);
            double new_stresses = 1 / r + stresses[s_next];
            if (new_stresses < stresses[start_stability]) {
                stresses[start_stability] = new_stresses;
                review_r[start_stability] = r;
            }
            for (int s_cur = s_next - 1; s_cur > start_stability; s_cur--) {
                r = exp(pow(s_cur, b) * ((float)s_next / (float)s_cur - 1) / a);
                new_stresses = 1 + r * stresses[s_next] + (1 - r) * stresses[start_stability];
                if (new_stresses < stresses[s_cur]) {
                    stresses[s_cur] = new_stresses;
                    review_r[s_cur] = r;
                }
            }
        }
        double diff = sum - accumulate(stresses, stresses + max_s, 0.0);
        if (diff < 0.1) {
            break;
        } else {
            sum = sum - diff;
        }
        printf("iter %d, diff %lf\n", i, diff);
    }
    FILE * f;
    f = fopen("30000-stress.csv", "w");
    for(int i=1; i<max_s; i++)
        fprintf(f, "%lf,%lf\n", stresses[i], review_r[i]);
    fclose(f);
    return 0;
}
