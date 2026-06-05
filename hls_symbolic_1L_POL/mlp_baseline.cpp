// mlp_baseline.cpp
// NeuSym-HLS — dense BL-MLP baseline, identical to the hybrid except fc3 is a
// real dense 64->10 layer instead of the symbolic head. Same fixed-point types,
// same MAC structure, so the only difference is the last layer's representation.
//
//   x[784] --fc1--> h1[128] --ReLU--> --fc2--> h2[64] --ReLU--> --fc3--> logits[10]
//
// Weights from export_baseline.py (baseline.pt -> baseline_params.h).

#include "mlp.h"
#include "baseline_params.h"   // W1,B1,W2,B2,W3,B3

#define UNROLL_F1 1
#define UNROLL_F2 1
#define UNROLL_F3 1

static inline feat_t relu_b(macc_t a) {
#pragma HLS INLINE
    return (a > (macc_t)0) ? (feat_t)a : (feat_t)0;
}

void mlp_baseline_top(const feat_t x[IN_DIM], logit_t logits[N_CLS]) {
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=logits
#pragma HLS INTERFACE ap_ctrl_hs port=return

    feat_t h1[H1_DIM];
    feat_t h2[H2_DIM];

    // fc1: 784 -> 128, ReLU
    FC1_OUT: for (int o = 0; o < H1_DIM; o++) {
        macc_t acc = (macc_t)B1[o];
        FC1_IN: for (int i = 0; i < IN_DIM; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UNROLL_F1
            acc += (macc_t)(W1[o][i] * x[i]);
        }
        h1[o] = relu_b(acc);
    }

    // fc2: 128 -> 64, ReLU
    FC2_OUT: for (int o = 0; o < H2_DIM; o++) {
        macc_t acc = (macc_t)B2[o];
        FC2_IN: for (int j = 0; j < H1_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UNROLL_F2
            acc += (macc_t)(W2[o][j] * h1[j]);
        }
        h2[o] = relu_b(acc);
    }

    // fc3: 64 -> 10, no activation (raw logits) — the DENSE replacement for
    // the symbolic head.
    FC3_OUT: for (int o = 0; o < N_CLS; o++) {
        macc_t acc = (macc_t)B3[o];
        FC3_IN: for (int j = 0; j < H2_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UNROLL_F3
            acc += (macc_t)(W3[o][j] * h2[j]);
        }
        logits[o] = (logit_t)acc;
    }
}
