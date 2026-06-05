// mlp_tb.cpp — csim testbench for the full hybrid network.
//
// To validate properly: dump a few golden samples from mlp_ref.py
//   python mlp_ref.py --dump 3
// then paste one input row into GOLDEN_X below and set GOLDEN_ARGMAX, so csim
// fails (returns nonzero) if the HLS argmax disagrees with the reference.

#include "mlp.h"
#include <cstdio>

int main() {
    feat_t x[IN_DIM];
    for (int i = 0; i < IN_DIM; ++i)
        x[i] = (feat_t)((i % 256) / 255.0f);   // dummy ramp; replace with real

    logit_t logits[N_CLS];
    mlp_top(x, logits);

    int argmax = 0; logit_t best = logits[0];
    for (int c = 0; c < N_CLS; ++c) {
        printf("logit[%d] = %f\n", c, (float)logits[c]);
        if (logits[c] > best) { best = logits[c]; argmax = c; }
    }
    printf("predicted class = %d\n", argmax);

    // golden check (uncomment after pasting a real sample)
    // const int GOLDEN_ARGMAX = 7;
    // if (argmax != GOLDEN_ARGMAX) { printf("MISMATCH\n"); return 1; }
    return 0;
}
