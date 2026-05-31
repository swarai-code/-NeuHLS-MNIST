#include "baseline_mlp.h"

static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

// ── FC1: 784 → 128, ReLU ─────────────────────────────────────────────────────
static void fc1_layer(
    float        x[IN_DIM],
    float        h1[H1_DIM],
    const float  w1[IN_DIM][H1_DIM],
    const float  b1[H1_DIM]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=w1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=x  cyclic factor=4

    FC1_J: for (int j = 0; j < H1_DIM; j++) {
#pragma HLS PIPELINE II=1
        float acc = b1[j];
        FC1_I: for (int i = 0; i < IN_DIM; i++) {
#pragma HLS UNROLL factor=4
            acc += x[i] * w1[i][j];
        }
        h1[j] = relu_f(acc);
    }
}

// ── FC2: 128 → 64, ReLU ──────────────────────────────────────────────────────
static void fc2_layer(
    float        h1[H1_DIM],
    float        h2[H2_DIM],
    const float  w2[H1_DIM][H2_DIM],
    const float  b2[H2_DIM]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=w2 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=h1 cyclic factor=4

    FC2_J: for (int j = 0; j < H2_DIM; j++) {
#pragma HLS PIPELINE II=1
        float acc = b2[j];
        FC2_I: for (int i = 0; i < H1_DIM; i++) {
#pragma HLS UNROLL factor=4
            acc += h1[i] * w2[i][j];
        }
        h2[j] = relu_f(acc);
    }
}

// ── FC3: 64 → 10 (no activation) ─────────────────────────────────────────────
static void fc3_layer(
    float        h2[H2_DIM],
    float        out[NUM_CLASSES],
    const float  w3[H2_DIM][NUM_CLASSES],
    const float  b3[NUM_CLASSES]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=h2 complete
#pragma HLS ARRAY_PARTITION variable=w3 complete dim=1

    FC3_J: for (int j = 0; j < NUM_CLASSES; j++) {
#pragma HLS PIPELINE II=1
        float acc = b3[j];
        FC3_I: for (int i = 0; i < H2_DIM; i++) {
#pragma HLS UNROLL complete
            acc += h2[i] * w3[i][j];
        }
        out[j] = acc;
    }
}

// ── Top-level ─────────────────────────────────────────────────────────────────
void baseline_mlp(
    float        x[IN_DIM],
    float        out[NUM_CLASSES],
    const float  w1[IN_DIM][H1_DIM],
    const float  b1[H1_DIM],
    const float  w2[H1_DIM][H2_DIM],
    const float  b2[H2_DIM],
    const float  w3[H2_DIM][NUM_CLASSES],
    const float  b3[NUM_CLASSES]
) {
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem0 depth=784
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=10
#pragma HLS INTERFACE m_axi port=w1  offset=slave bundle=gmem2 depth=100352
#pragma HLS INTERFACE m_axi port=b1  offset=slave bundle=gmem3 depth=128
#pragma HLS INTERFACE m_axi port=w2  offset=slave bundle=gmem4 depth=8192
#pragma HLS INTERFACE m_axi port=b2  offset=slave bundle=gmem5 depth=64
#pragma HLS INTERFACE m_axi port=w3  offset=slave bundle=gmem6 depth=640
#pragma HLS INTERFACE m_axi port=b3  offset=slave bundle=gmem7 depth=10
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW

    float h1[H1_DIM];
    float h2[H2_DIM];
#pragma HLS ARRAY_PARTITION variable=h1 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=h2 complete

    fc1_layer(x, h1, w1, b1);
    fc2_layer(h1, h2, w2, b2);
    fc3_layer(h2, out, w3, b3);
}
