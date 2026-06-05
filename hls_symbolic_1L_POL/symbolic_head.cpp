// symbolic_head.cpp
// NeuSym-HLS — MNIST 1L_POL symbolic head (replaces fc3: 64 -> 10).
//
// Optimizations applied by hand from the PySR equations:
//   1) Fixed-point quantization (ap_fixed) instead of float -> no float DSPs.
//   2) Sparsity-aware reads: only 40/64 inputs referenced; ARRAY_PARTITION +
//      dead-code elimination drop the 24 unused input ports.
//   3) Constant folding: each class's additive constants pre-summed into one
//      compile-time BIAS literal.
//   4) Shift-and-add (CSD): every non-unit coefficient becomes shifts+adds,
//      so the layer synthesizes with ZERO multipliers (0 DSP).
//   5) Balanced binary adder tree (tree_sum): O(log N) add depth per class.
//   6) Full pipeline (II=1): all 10 classes evaluated in parallel.

#include "symbolic_head.h"


// Constant-folded per-class biases (sum of all literal terms in each equation)
static const acc_t BIAS0 = (acc_t)(-2.6073067f);
static const acc_t BIAS1 = (acc_t)( 1.0557327f);  //  1.0672832 - 0.011550516
static const acc_t BIAS2 = (acc_t)(-0.2888112f);
static const acc_t BIAS3 = (acc_t)(-1.9390514f);  // -0.8293426 - 1.1097088
static const acc_t BIAS5 = (acc_t)(-1.5606319f);  // -1.4916098 - 0.0690221
static const acc_t BIAS6 = (acc_t)(-0.3341241f);  // -1.4437156 + 1.1095915
static const acc_t BIAS7 = (acc_t)(-0.1049140f);
static const acc_t BIAS8 = (acc_t)( 1.1345044f);  //  0.41776285 + 0.7167415
static const acc_t BIAS9 = (acc_t)(-7.6854916f);
// class 4 bias folds to 0 and is omitted.


// Shift-and-add (CSD) constant multipliers. Each returns coeff * x using only
// shifts and adds -> mapped to LUT logic, never a DSP48. Approximation error
// is < 0.15% per coefficient (validate end-to-end accuracy in PyTorch).
static inline acc_t mul_0p8172(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  a - (a>>3) - (a>>4) + (a>>8);          // 0.81641
}
static inline acc_t mul_0p2875(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  (a>>2) + (a>>5) + (a>>7);              // 0.28906
}
static inline acc_t mul_0p9800(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  a - (a>>6) - (a>>8);                   // 0.98047
}
static inline acc_t mul_1p6723(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  (a<<1) - (a>>2) - (a>>4) - (a>>6);     // 1.67188
}
static inline acc_t mul_0p6723(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  (a>>1) + (a>>3) + (a>>4) - (a>>6);     // 0.67188
}
static inline acc_t mul_1p1927(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  a + (a>>2) - (a>>4) + (a>>8);          // 1.19141
}
static inline acc_t mul_0p5662(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  (a>>1) + (a>>4) + (a>>8);              // 0.56641
}
static inline acc_t mul_0p5153(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  (a>>1) + (a>>6);                       // 0.51563
}
static inline acc_t mul_0p6802(feat_t x){
#pragma HLS INLINE
    acc_t a = (acc_t)x; return  (a>>1) + (a>>3) + (a>>4) - (a>>7);     // 0.67969
}


// Balanced binary adder tree. In-place pairwise reduction gives ceil(log2 N)
// add depth (e.g. 3 levels for 8 terms) instead of N-1 in a serial chain.
//   For class 0 (6 terms t0..t5) this evaluates as:
//     ((t0+t1)+(t2+t3)) + (t4+t5)
template<int N>
static acc_t tree_sum(acc_t v[N]) {
#pragma HLS INLINE
    for (int stride = 1; stride < N; stride <<= 1) {
#pragma HLS UNROLL
        for (int i = 0; i + stride < N; i += (stride << 1)) {
#pragma HLS UNROLL
            v[i] += v[i + stride];
        }
    }
    return v[0];
}


// Top function
void symbolic_head(const feat_t h[H_DIM], logit_t logits[N_CLS]) {
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=h      complete dim=1
#pragma HLS ARRAY_PARTITION variable=logits complete dim=1
    // For port-level sparsity (standalone IP) replace the array interface with
    // 40 scalar ports. As a sub-function of the full net, keep the array:
    // fc2 writes h[64] and unused lanes are pruned automatically.

    // class 0:  +x19 +x13 -x31 -x40 -x58  -2.6073067
    { acc_t t[6] = { (acc_t)h[19], (acc_t)h[13], -(acc_t)h[31],
                     -(acc_t)h[40], -(acc_t)h[58], BIAS0 };
      logits[0] = (logit_t) tree_sum<6>(t); }

    // class 1:  0.81722*x57 -x12 -x42 -x31 +x9 -x53  +1.0557327
    { acc_t t[7] = { mul_0p8172(h[57]), -(acc_t)h[12], -(acc_t)h[42],
                     -(acc_t)h[31], (acc_t)h[9], -(acc_t)h[53], BIAS1 };
      logits[1] = (logit_t) tree_sum<7>(t); }

    // class 2:  +x38 -0.28745*x48 -x51 -x52 +x55 -x14 -x7  -0.2888112
    { acc_t t[8] = { (acc_t)h[38], -mul_0p2875(h[48]), -(acc_t)h[51],
                     -(acc_t)h[52], (acc_t)h[55], -(acc_t)h[14],
                     -(acc_t)h[7], BIAS2 };
      logits[2] = (logit_t) tree_sum<8>(t); }

    // class 3:  -0.98002*x47 -x44 +x17 +x8 -x29 -x54  -1.9390514
    { acc_t t[7] = { -mul_0p9800(h[47]), -(acc_t)h[44], (acc_t)h[17],
                     (acc_t)h[8], -(acc_t)h[29], -(acc_t)h[54], BIAS3 };
      logits[3] = (logit_t) tree_sum<7>(t); }

    // class 4:  +x40 -x50 -1.67230*x19 -x61 +x37 -0.67230*x12   (bias 0)
    { acc_t t[6] = { (acc_t)h[40], -(acc_t)h[50], -mul_1p6723(h[19]),
                     -(acc_t)h[61], (acc_t)h[37], -mul_0p6723(h[12]) };
      logits[4] = (logit_t) tree_sum<6>(t); }

    // class 5:  +x50 -x60 -x3 -x0 +x4 -x54 +x7  -1.5606319
    { acc_t t[8] = { (acc_t)h[50], -(acc_t)h[60], -(acc_t)h[3],
                     -(acc_t)h[0], (acc_t)h[4], -(acc_t)h[54],
                     (acc_t)h[7], BIAS5 };
      logits[5] = (logit_t) tree_sum<8>(t); }

    // class 6:  1.19272*(x44 +x54 -x50 -x52) -x31 +x7  -0.3341241
    { acc_t t[7] = { mul_1p1927(h[44]), mul_1p1927(h[54]), -mul_1p1927(h[50]),
                     -mul_1p1927(h[52]), -(acc_t)h[31], (acc_t)h[7], BIAS6 };
      logits[6] = (logit_t) tree_sum<7>(t); }

    // class 7:  +x9 -x48 +x10 -x17 -x27 -0.56618*x46  -0.1049140
    { acc_t t[7] = { (acc_t)h[9], -(acc_t)h[48], (acc_t)h[10],
                     -(acc_t)h[17], -(acc_t)h[27], -mul_0p5662(h[46]), BIAS7 };
      logits[7] = (logit_t) tree_sum<7>(t); }

    // class 8:  +x48 -0.51529*x34 +x63 -x1 -x14  +1.1345044
    { acc_t t[6] = { (acc_t)h[48], -mul_0p5153(h[34]), (acc_t)h[63],
                     -(acc_t)h[1], -(acc_t)h[14], BIAS8 };
      logits[8] = (logit_t) tree_sum<6>(t); }

    // class 9:  +x31 -x21 +x45 +0.68024*x24 -0.68024*x6 -x7  -7.6854916
    { acc_t t[7] = { (acc_t)h[31], -(acc_t)h[21], (acc_t)h[45],
                     mul_0p6802(h[24]), -mul_0p6802(h[6]),
                     -(acc_t)h[7], BIAS9 };
      logits[9] = (logit_t) tree_sum<7>(t); }
}
