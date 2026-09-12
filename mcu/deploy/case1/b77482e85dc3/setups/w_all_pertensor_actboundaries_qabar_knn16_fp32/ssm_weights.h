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
    const int8_t *w_q;
    const float *w_scale;
} ssm_norm_weights_t;

typedef struct {
    const int8_t *in_proj_w_q;
    const float *in_proj_w_scale;
    const int8_t *conv_w_q;
    const float *conv_w_scale;
    const int8_t *out_proj_w_q;
    const float *out_proj_w_scale;
    const int8_t *conv_b_q;
    const float *conv_b_scale;
    const int8_t *D_q;
    const float *D_scale;
    const int8_t *A_bar_q;
    const float *A_bar_scale;
    const int8_t *B_bar_q;
    const float *B_bar_scale;
    const int8_t *C_q;
    const float *C_scale;
    float u_scale;
    float z_scale;
    float y_gated_scale;
    float block_out_scale;
    ssm_norm_weights_t norm_w;
} ssm_block_weights_t;

#define SSM_IN_PROJ_W(w, r, c) ((float)(w)->in_proj_w_q[(r) * SSM_D_MODEL + (c)] * (w)->in_proj_w_scale[0])
#define SSM_CONV_W(w, r, c) ((float)(w)->conv_w_q[(r) * SSM_D_CONV + (c)] * (w)->conv_w_scale[0])
#define SSM_OUT_PROJ_W(w, r, c) ((float)(w)->out_proj_w_q[(r) * SSM_D_INNER + (c)] * (w)->out_proj_w_scale[0])
#define SSM_CONV_B(w, i) ((float)(w)->conv_b_q[(i)] * (w)->conv_b_scale[0])
#define SSM_D_PARAM(w, i) ((float)(w)->D_q[(i)] * (w)->D_scale[0])
#define SSM_A_BAR_BAKED(w, r, c) ((float)(w)->A_bar_q[(r) * SSM_D_STATE + (c)] * (w)->A_bar_scale[0])
#define SSM_B_BAR_BAKED(w, r, c) ((float)(w)->B_bar_q[(r) * SSM_D_STATE + (c)] * (w)->B_bar_scale[0])
#define SSM_C(w, i) ((float)(w)->C_q[(i)] * (w)->C_scale[0])
#define SSM_DELTA(w, c) (0.0f)
#define SSM_A_BAR(w, c, n, d) ((void)(d), SSM_A_BAR_BAKED((w), (c), (n)))
#define SSM_B_BAR(w, c, n, d) ((void)(d), SSM_B_BAR_BAKED((w), (c), (n)))
#define SSM_QUANT_U(w, val) (ssm_quant_dequant((val), (w)->u_scale))
#define SSM_QUANT_Z(w, val) (ssm_quant_dequant((val), (w)->z_scale))
#define SSM_QUANT_Y_GATED(w, val) (ssm_quant_dequant((val), (w)->y_gated_scale))
#define SSM_QUANT_BLOCK_OUT(w, val) (ssm_quant_dequant((val), (w)->block_out_scale))
#define SSM_QUANT_FINAL_NORM(val) (ssm_quant_dequant((val), ssm_final_norm_scale))
#define SSM_NORM_W(nw, i) ((float)(nw)->w_q[(i)] * (nw)->w_scale[0])

extern const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS];
extern const ssm_norm_weights_t ssm_final_norm_w;
extern const float ssm_final_norm_scale;

extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
