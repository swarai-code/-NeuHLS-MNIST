// symbolic_head.h
// NeuSym-HLS — MNIST 1L_POL symbolic head (replaces fc3: 64 -> 10).
// Hand-optimized: ap_fixed quantization, sparsity-aware reads, constant
// folding, shift-and-add (CSD) coefficients, balanced adder tree.

#ifndef SYMBOLIC_HEAD_H
#define SYMBOLIC_HEAD_H

#include <ap_fixed.h>

// Inputs are post-ReLU activations of fc2 (h2, 64-dim).
typedef ap_fixed<16, 8>  feat_t;   // layer inputs / features
typedef ap_fixed<24, 10> acc_t;    // internal accumulation + shift-add work
typedef ap_fixed<16, 8>  logit_t;  // 10 output logits

#define H_DIM  64
#define N_CLS  10

// Top function: full 64-wide input array in, 10 logits out.
// Only 40 of the 64 inputs are referenced; the rest are pruned in synthesis.
void symbolic_head(const feat_t h[H_DIM], logit_t logits[N_CLS]);

#endif // SYMBOLIC_HEAD_H
