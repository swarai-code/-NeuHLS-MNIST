// mlp_baseline_tb.cpp — smoke test for the dense baseline.
#include "mlp.h"
#include <cstdio>
void mlp_baseline_top(const feat_t x[IN_DIM], logit_t logits[N_CLS]);

int main() {
    feat_t x[IN_DIM];
    for (int i = 0; i < IN_DIM; ++i)
        x[i] = (feat_t)(((i % 256) / 255.0f - 0.1307f) / 0.3081f); // normalized ramp
    logit_t logits[N_CLS];
    mlp_baseline_top(x, logits);
    int am = 0; logit_t best = logits[0];
    for (int c = 0; c < N_CLS; ++c) {
        printf("logit[%d] = %f\n", c, (float)logits[c]);
        if (logits[c] > best) { best = logits[c]; am = c; }
    }
    printf("predicted class = %d\n", am);
    return 0;
}
