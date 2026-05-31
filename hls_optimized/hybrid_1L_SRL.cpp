#include "hybrid_1L_SRL.h"
#include <cmath>

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

// ── Symbolic layer: replaces fc3 (64 → 10) ───────────────────────────────────
// Equations from PySR 1L_SRL experiment (best accuracy: 97.87% after fine-tune)
static void symbolic_layer(float h2[H2_DIM], float out[NUM_CLASSES]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=h2 complete

    // class 0  loss=8.54  complexity=17
    out[0] = ((((h2[33] - h2[2]) * 0.5676503f) - (h2[29] + h2[28]))
              - (h2[8] - h2[43])) + 0.11564218f;

    // class 1  loss=14.13  complexity=17
    out[1] = ((((h2[2] + h2[42]) - h2[31]) * 0.5259374f)
              - (h2[55] - 0.18107876f)) + (h2[57] - h2[49]);

    // class 2  loss=14.09  complexity=19
    out[2] = (((h2[2] - h2[44]) - h2[10]) - 0.381145f)
             + ((h2[20] - h2[15]) - ((h2[59] + 1.6749516f) * 0.35072204f));

    // class 3  loss=9.96  complexity=19
    out[3] = (h2[17] + h2[7])
             + ((((h2[8] - h2[47]) - (h2[23] - 1.376536f)) * 0.7248438f)
                - (h2[29] + h2[44]));

    // class 4  loss=10.50  complexity=19
    out[4] = h2[40]
             + ((h2[59] * 0.5853861f)
                + (h2[45] - ((h2[38] + (h2[49] * 1.3468009f)) + 1.3310196f)));

    // class 5  loss=12.68  complexity=19  (uses relu)
    out[5] = ((h2[7] + h2[30]) * 0.69933736f)
             - (relu_f(h2[23] - h2[50]) - ((h2[4] - h2[3]) - h2[0]));

    // class 6  loss=13.08  complexity=19
    out[6] = h2[44]
             + (h2[42] + (((h2[23] - h2[31]) - 8.358079f)
                          - (h2[50] + ((h2[0] - h2[46]) * 0.66487896f))));

    // class 7  loss=11.89  complexity=15
    out[7] = (-2.552997f - h2[48])
             + (h2[10] + ((h2[16] - h2[61]) - ((h2[42] + h2[46]) + 0.55382746f)));

    // class 8  loss=10.07  complexity=19
    out[8] = h2[63]
             + (h2[48] - ((h2[16] * 0.40036032f)
                          + ((h2[14] - 0.79678404f) + (h2[1] * 0.8601214f))));

    // class 9  loss=12.82  complexity=15
    out[9] = (h2[31] + (h2[54] - (h2[34] - 1.5835959f)))
             + (((0.12684631f - h2[6]) - h2[21]) - h2[7]);
}

// ── Top-level ─────────────────────────────────────────────────────────────────
void hybrid_1L_SRL(
    float        x[IN_DIM],
    float        out[NUM_CLASSES],
    const float  w1[IN_DIM][H1_DIM],
    const float  b1[H1_DIM],
    const float  w2[H1_DIM][H2_DIM],
    const float  b2[H2_DIM]
) {
#pragma HLS INTERFACE m_axi port=x   offset=slave bundle=gmem0 depth=784
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=10
#pragma HLS INTERFACE m_axi port=w1  offset=slave bundle=gmem2 depth=100352
#pragma HLS INTERFACE m_axi port=b1  offset=slave bundle=gmem3 depth=128
#pragma HLS INTERFACE m_axi port=w2  offset=slave bundle=gmem4 depth=8192
#pragma HLS INTERFACE m_axi port=b2  offset=slave bundle=gmem5 depth=64
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW

    float h1[H1_DIM];
    float h2[H2_DIM];
#pragma HLS ARRAY_PARTITION variable=h1 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=h2 complete

    fc1_layer(x, h1, w1, b1);
    fc2_layer(h1, h2, w2, b2);
    symbolic_layer(h2, out);
}
