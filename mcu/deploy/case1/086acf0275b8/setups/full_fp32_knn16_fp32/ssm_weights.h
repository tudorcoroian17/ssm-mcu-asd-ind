#pragma once

#include <stdint.h>
#include <math.h>

#define SSM_D_MODEL 64
#define SSM_D_STATE 8
#define SSM_D_INNER 128
#define SSM_D_CONV 4
#define SSM_DT_RANK 4
#define SSM_N_LAYERS 2

typedef struct {
    const float *w;
} ssm_norm_weights_t;

typedef struct {
    const float *in_proj_w;
    const float *conv_w;
    const float *x_proj_w;
    const float *dt_proj_w;
    const float *out_proj_w;
    const float *conv_b;
    const float *dt_proj_b;
    const float *D;
    const float *A;
    ssm_norm_weights_t norm_w;
} ssm_block_weights_t;

#define SSM_IN_PROJ_W(w, r, c) ((w)->in_proj_w[(r) * SSM_D_MODEL + (c)])
#define SSM_CONV_W(w, r, c) ((w)->conv_w[(r) * SSM_D_CONV + (c)])
#define SSM_X_PROJ_W(w, r, c) ((w)->x_proj_w[(r) * SSM_D_INNER + (c)])
#define SSM_DT_PROJ_W(w, r, c) ((w)->dt_proj_w[(r) * SSM_DT_RANK + (c)])
#define SSM_OUT_PROJ_W(w, r, c) ((w)->out_proj_w[(r) * SSM_D_INNER + (c)])
#define SSM_CONV_B(w, i) ((w)->conv_b[(i)])
#define SSM_DT_PROJ_B(w, i) ((w)->dt_proj_b[(i)])
#define SSM_D_PARAM(w, i) ((w)->D[(i)])
#define SSM_A(w, r, c) ((w)->A[(r) * SSM_D_STATE + (c)])
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
