// symbolic_head_tb.cpp — minimal smoke test for csim.
// Replace the input vector with a real fc2 activation row + compare argmax
// against your PyTorch hybrid model for proper validation.

#include "symbolic_head.h"
#include <cstdio>

int main() {
    feat_t h[H_DIM];
    for (int i = 0; i < H_DIM; ++i) h[i] = (feat_t)(0.1f * ((i % 7) - 3)); // dummy

    logit_t logits[N_CLS];
    symbolic_head(h, logits);

    int argmax = 0;
    logit_t best = logits[0];
    for (int c = 0; c < N_CLS; ++c) {
        printf("logit[%d] = %f\n", c, (float)logits[c]);
        if (logits[c] > best) { best = logits[c]; argmax = c; }
    }
    printf("predicted class = %d\n", argmax);
    return 0;  // return nonzero to fail csim if you add a golden check
}
