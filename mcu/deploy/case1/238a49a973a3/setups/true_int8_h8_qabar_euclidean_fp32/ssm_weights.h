#pragma once

#include <stdint.h>

/* CLASSIC (selective=False), true int8 arithmetic. No SSM_DT_RANK and no
 * softplus LUT: delta = softplus(dt) is a constant on this branch, folded
 * into delta_q (qab) or into A_bar/B_bar (qabar) at export time. */

#define SSM_D_MODEL 64
#define SSM_D_STATE 16
#define SSM_D_INNER 64
#define SSM_D_CONV 4
#define SSM_N_LAYERS 2

/* One layer's weights, scales and LUTs. Shape depends on the recurrence
 * form; ssm_backbone.c never names these members directly, only through the
 * SSM_TI_* macros below, so its text is identical for qab and qabar. */
typedef struct {
	const int8_t *in_proj_u_w_q;
	const int8_t *in_proj_z_w_q;
	float in_proj_w_scale;

	const int8_t *conv_w_q;
	float conv_w_scale;
	const float *conv_b;

	const int8_t *out_proj_w_q;
	float out_proj_w_scale;

	const int8_t *C_q;
	const float *D;
	const float *norm_w;

	const int8_t *A_bar_q;
	const int8_t *B_bar_q;

	float s_norm_out;
	float s_conv_out;
	float s_u_post_conv_silu;
	float s_z_gate;
	float s_B;
	float s_C;
	float s_delta;
	float s_A_bar;
	float s_B_bar;
	float s_h;
	float s_y_scan;
	float s_y_gated;
	float s_block_output;

	const int8_t *lut_silu_conv;
	const int8_t *lut_silu_z;
	float s_silu_z;
} ssm_true_int8_layer_t;

/* s_delta and s_B are emitted under both forms so this struct and
 * TI_SCALE_FIELD_MAP_CLASSIC stay uniform; under qabar nothing reads them. */

#define SSM_TI_DELTA_REAL(w, c) (0.0f)
#define SSM_TI_A_BAR_Q(w, c, n, d) ((void) (d), (w)->A_bar_q[(c) * SSM_D_STATE + (n)])
#define SSM_TI_B_BAR_Q(w, c, n, d) ((void) (d), (w)->B_bar_q[(c) * SSM_D_STATE + (n)])

/* Deliberately does NOT #include "ssm_backbone.h" -- that header includes
 * THIS file for the macros, h-width flag, and this struct type. */

extern const ssm_true_int8_layer_t ssm_layers[SSM_N_LAYERS];
extern const float ssm_final_norm_w[SSM_D_MODEL];
extern const float ssm_final_norm_scale;
extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
