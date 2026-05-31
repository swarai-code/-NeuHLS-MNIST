#pragma once

#define IN_DIM      784
#define H1_DIM      128
#define H2_DIM       64
#define NUM_CLASSES  10

// Baseline MNIST MLP: 784 → 128 (ReLU) → 64 (ReLU) → 10
void baseline_mlp(
    float        x[IN_DIM],
    float        out[NUM_CLASSES],
    const float  w1[IN_DIM][H1_DIM],
    const float  b1[H1_DIM],
    const float  w2[H1_DIM][H2_DIM],
    const float  b2[H2_DIM],
    const float  w3[H2_DIM][NUM_CLASSES],
    const float  b3[NUM_CLASSES]
);
