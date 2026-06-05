// mlp.h
// NeuSym-HLS — full MNIST hybrid network:
//   fc1 (784->128, ReLU) + fc2 (128->64, ReLU) + symbolic head (64->10).
// fc1/fc2 are quantized fixed-point MACs; fc3 is the shift-add symbolic layer.

#ifndef MLP_H
#define MLP_H

#include "symbolic_head.h"   // brings in feat_t, acc_t, logit_t, N_CLS

// Layer dimensions
#define IN_DIM  784
#define H1_DIM  128
#define H2_DIM  64
// (H2_DIM must equal H_DIM=64 from symbolic_head.h)

// Quantized weight type for fc1/fc2. ap_fixed<8,2>: range [-2, ~2), res 2^-6.
// MNIST MLP weights are small, so 2 integer bits is usually plenty — widen if
// the export script reports clipping.
typedef ap_fixed<8, 2>   w_t;
// MAC accumulator: 784 products need headroom. ap_fixed<32,16> = range +/-32768.
typedef ap_fixed<32, 16> macc_t;

// Top function: 784 input pixels -> 10 logits.
void mlp_top(const feat_t x[IN_DIM], logit_t logits[N_CLS]);

#endif // MLP_H
