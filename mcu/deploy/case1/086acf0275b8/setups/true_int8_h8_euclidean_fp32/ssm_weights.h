#pragma once

#include <stdint.h>

#define SSM_D_MODEL 64
#define SSM_D_STATE 8
#define SSM_D_INNER 128
#define SSM_D_CONV 4
#define SSM_DT_RANK 4
#define SSM_N_LAYERS 2

/* One layer's weights, scales, and LUTs -- true int8 arithmetic scheme.
 * Fixed shape regardless of h-width (only the VALUES of s_h and
 * ssm_backbone.h's ssm_h_t typedef change between the two exports). */
typedef struct {
	const int8_t *in_proj_u_w_q;
	const int8_t *in_proj_z_w_q;
	float in_proj_w_scale;

	const int8_t *conv_w_q;
	float conv_w_scale;
	const float *conv_b;

	const int8_t *x_proj_w_q;
	float x_proj_w_scale;

	const int8_t *dt_proj_w_q;
	float dt_proj_w_scale;
	const float *dt_proj_b;

	const int8_t *out_proj_w_q;
	float out_proj_w_scale;

	const float *A;
	const float *D;
	const float *norm_w;

	float s_norm_out;
	float s_conv_out;
	float s_u_post_conv_silu;
	float s_z_gate;
	float s_x_proj_delta_low_out;
	float s_B;
	float s_C;
	float s_dt_proj_out;
	float s_delta;
	float s_A_bar;
	float s_B_bar;
	float s_h;
	float s_y_scan;
	float s_y_gated;
	float s_block_output;

	const int8_t *lut_softplus;
	const int8_t *lut_silu_conv;
	const int8_t *lut_silu_z;
	float s_silu_z;
} ssm_true_int8_layer_t;

/* Deliberately does NOT #include "ssm_backbone.h" -- ssm_backbone.h
 * includes THIS file for the macros, h-width flag, and this struct type;
 * including it back would be circular. ssm_weights.c needs only this. */

extern const ssm_true_int8_layer_t ssm_layers[SSM_N_LAYERS];
extern const float ssm_final_norm_w[SSM_D_MODEL];
extern const float ssm_final_norm_scale;
extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
