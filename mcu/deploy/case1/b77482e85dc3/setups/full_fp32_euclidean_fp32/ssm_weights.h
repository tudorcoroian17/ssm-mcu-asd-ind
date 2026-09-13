#pragma once

#include <stdint.h>
#include <math.h>

/* CLASSIC (selective=False) backbone. No SSM_DT_RANK: this branch has no
 * x_proj/dt_proj, so there is no low-rank delta path to size. */

#define SSM_D_MODEL 64
#define SSM_D_STATE 8
#define SSM_D_INNER 128
#define SSM_D_CONV 4
#define SSM_N_LAYERS 2

typedef struct {
    const float *w;
} ssm_norm_weights_t;

typedef struct {
    const float *in_proj_w;
    const float *conv_w;
    const float *out_proj_w;
    const float *conv_b;
    const float *D;
    const float *A;
    const float *dt;
    const float *B;
    const float *C;
    ssm_norm_weights_t norm_w;
} ssm_block_weights_t;

#define SSM_IN_PROJ_W(w, r, c) ((w)->in_proj_w[(r) * SSM_D_MODEL + (c)])
#define SSM_CONV_W(w, r, c) ((w)->conv_w[(r) * SSM_D_CONV + (c)])
#define SSM_OUT_PROJ_W(w, r, c) ((w)->out_proj_w[(r) * SSM_D_INNER + (c)])
#define SSM_CONV_B(w, i) ((w)->conv_b[(i)])
#define SSM_D_PARAM(w, i) ((w)->D[(i)])
#define SSM_A(w, r, c) ((w)->A[(r) * SSM_D_STATE + (c)])
#define SSM_DT(w, i) ((w)->dt[(i)])
#define SSM_B(w, i) ((w)->B[(i)])
#define SSM_C(w, i) ((w)->C[(i)])
#define SSM_DELTA(w, c) ssm_softplus(SSM_DT((w), (c)))
#define SSM_A_BAR(w, c, n, d) ssm_euler_abar((d) * SSM_A((w), (c), (n)))
#define SSM_B_BAR(w, c, n, d) ((d) * SSM_B((w), (n)))
#define SSM_QUANT_U(w, val) (val)
#define SSM_QUANT_Z(w, val) (val)
#define SSM_QUANT_Y_GATED(w, val) (val)
#define SSM_QUANT_BLOCK_OUT(w, val) (val)
#define SSM_QUANT_FINAL_NORM(val) (val)
#define SSM_NORM_W(nw, i) ((nw)->w[(i)])

extern const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS];
extern const ssm_norm_weights_t ssm_final_norm_w;


extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
