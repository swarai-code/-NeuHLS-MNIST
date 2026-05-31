#pragma once

#define IN_DIM      784
#define H1_DIM      128
#define H2_DIM       64
#define NUM_CLASSES  10

// Hybrid MNIST MLP: 784 → 128 (ReLU) → 64 (ReLU) → symbolic(10)
// fc3 replaced by 1L_SRL symbolic expressions from PySR
void hybrid_1L_SRL(
    float        x[IN_DIM],
    float        out[NUM_CLASSES],
    const float  w1[IN_DIM][H1_DIM],
    const float  b1[H1_DIM],
    const float  w2[H1_DIM][H2_DIM],
    const float  b2[H2_DIM]
);
