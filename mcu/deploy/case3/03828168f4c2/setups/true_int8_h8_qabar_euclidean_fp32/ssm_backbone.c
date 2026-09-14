#include "ssm_backbone.h"

#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------------
 * CLASSIC (selective=False), true int8 arithmetic. Mirrors
 * true_int8_sim_classic.py's quantized_block_step_classic line for line,
 * which in turn mirrors the selective quantized_block_step minus the
 * x_proj / split / dt_proj / softplus-LUT section.
 *
 * The recurrence coefficients arrive through SSM_TI_DELTA_REAL,
 * SSM_TI_A_BAR_Q and SSM_TI_B_BAR_Q, defined in the GENERATED
 * ssm_weights.h. Under qab those expand to the live discretize math
 * (requantizing A_bar and B_bar every frame, exactly as the selective
 * backbone does); under qabar they read baked int8 arrays and discard
 * delta_c. This file's text is identical either way.
 *
 * Quantization primitives are unchanged from the selective true-int8
 * source, deliberately, so the two can be diffed.
 * --------------------------------------------------------------------- */

static inline int8_t ssm_requantize(float real, float scale) {
	float q = roundf(real / scale);
	if (q > 127.0f) q = 127.0f;
	if (q < -127.0f) q = -127.0f;
	return (int8_t) q;
}

static void ssm_quantize_arr(const float *real, float scale, int n, int8_t *out_q) {
	for (int i = 0; i < n; i++) {
		out_q[i] = ssm_requantize(real[i], scale);
	}
}

static inline int8_t ssm_apply_lut(const int8_t *lut, int8_t x_q) {
	return lut[(int) x_q + 128];
}

/* 	The euler step from ssm_block.py's discretize(). Used only by the qab
 * 	macro expansion; unused (and harmless) in a qabar build. */
static inline float ssm_euler_abar(float deltaA) {
	if (deltaA < -1.9f) {
		deltaA = -1.9f;
	}
	return 1.0f + deltaA;
}

/* Row-major int8 matvec: x_q (in_dim,) int8, w_q (out_dim, in_dim) int8 ->
 * out_real (out_dim,) real units. int32 accumulation is exact and safe
 * here: max |acc| <= 127*127*in_dim, and the largest in_dim in this model
 * is SSM_D_INNER, so for d_inner = 128 that is ~2.06e6, well inside
 * int32's ~2.1e9 range. */
static void ssm_qlinear_real(const int8_t *x_q, const int8_t *w_q,
		int out_dim, int in_dim, float x_scale, float w_scale,
		const float *bias, float *out_real) {
	for (int o = 0; o < out_dim; o++) {
		const int8_t *w_row = &w_q[o * in_dim];
		int32_t acc = 0;
		for (int i = 0; i < in_dim; i++) {
			acc += (int32_t) w_row[i] * (int32_t) x_q[i];
		}
		float real = (float) acc * x_scale * w_scale;
		if (bias != NULL) {
			real += bias[o];
		}
		out_real[o] = real;
	}
}

static void ssm_rmsnorm(const float *x, const float *weight, float *out) {
	float ss = 0.0f;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		ss += x[i] * x[i];
	}
	float inv_rms = 1.0f / sqrtf(ss / (float) SSM_D_MODEL + 1e-5f);
	for (int i = 0; i < SSM_D_MODEL; i++) {
		out[i] = x[i] * inv_rms * weight[i];
	}
}

/* One block's forward pass, true int8 arithmetic throughout except:
 *   - RMSNorm's sum-of-squares/rsqrt (done by the caller, before this
 *     function)
 *   - biases, added in real units at each op's rescale point
 *   - A itself under qab, a static precomputed real array
 * Every other value crossing a function boundary here is int8 (or, for h,
 * ssm_h_t), with its scale tracked separately. */
static void ssm_block_step_true_int8(
		const ssm_true_int8_layer_t *w,
		ssm_h_t h[SSM_D_INNER][SSM_D_STATE],
		int8_t conv_hist_q[SSM_D_INNER][SSM_D_CONV - 1],
		const int8_t *x_norm_q,
		int8_t *block_out_q) {

	float u_raw_real[SSM_D_INNER];
	float z_real[SSM_D_INNER];
	int8_t u_raw_q[SSM_D_INNER];
	int8_t z_q[SSM_D_INNER];
	int8_t u_q[SSM_D_INNER];

	/* in_proj, split into u_raw and z. Requantized against s_conv_out /
	 * s_z_gate respectively -- NOT a shared scale. */
	ssm_qlinear_real(x_norm_q, w->in_proj_u_w_q, SSM_D_INNER, SSM_D_MODEL,
			w->s_norm_out, w->in_proj_w_scale, NULL, u_raw_real);
	ssm_qlinear_real(x_norm_q, w->in_proj_z_w_q, SSM_D_INNER, SSM_D_MODEL,
			w->s_norm_out, w->in_proj_w_scale, NULL, z_real);
	for (int c = 0; c < SSM_D_INNER; c++) {
		u_raw_q[c] = ssm_requantize(u_raw_real[c], w->s_conv_out);
		z_q[c] = ssm_requantize(z_real[c], w->s_z_gate);
	}

	/* Depthwise causal conv: tap 0 = oldest (t-3) .. tap D_CONV-1 =
	 * current. conv_hist stores PRE-conv u_raw_q values, oldest-first, so
	 * all taps share s_conv_out's scale domain. */
	for (int c = 0; c < SSM_D_INNER; c++) {
		const int8_t *w_row = &w->conv_w_q[c * SSM_D_CONV];
		int32_t acc = 0;
		for (int k = 0; k < SSM_D_CONV - 1; k++) {
			acc += (int32_t) w_row[k] * (int32_t) conv_hist_q[c][k];
		}
		acc += (int32_t) w_row[SSM_D_CONV - 1] * (int32_t) u_raw_q[c];

		float conv_real = (float) acc * w->s_conv_out * w->conv_w_scale + w->conv_b[c];
		int8_t conv_out_q = ssm_requantize(conv_real, w->s_conv_out);
		u_q[c] = ssm_apply_lut(w->lut_silu_conv, conv_out_q);

		for (int k = 0; k < SSM_D_CONV - 2; k++) {
			conv_hist_q[c][k] = conv_hist_q[c][k + 1];
		}
		conv_hist_q[c][SSM_D_CONV - 2] = u_raw_q[c];
	}

	/* Recurrence + output, fused per channel. No x_proj, no dt_proj, no
	 * softplus LUT: delta, B and C are static on this branch. */
	float block_out_real[SSM_D_MODEL];
	int8_t y_gated_q[SSM_D_INNER];

	for (int c = 0; c < SSM_D_INNER; c++) {
		const float delta_c = SSM_TI_DELTA_REAL(w, c);

		int32_t y_scan_acc = 0;
		for (int n = 0; n < SSM_D_STATE; n++) {
			int8_t A_bar_q = SSM_TI_A_BAR_Q(w, c, n, delta_c);
			int8_t B_bar_q = SSM_TI_B_BAR_Q(w, c, n, delta_c);

			float Ah_real = (float) ((int32_t) A_bar_q * (int32_t) h[c][n])
					* w->s_A_bar * w->s_h;
			float Bu_real = (float) ((int32_t) B_bar_q * (int32_t) u_q[c])
					* w->s_B_bar * w->s_u_post_conv_silu;
			float new_h_real = Ah_real + Bu_real;

			float new_h_q_f = roundf(new_h_real / w->s_h);
			if (new_h_q_f > (float) SSM_H_CLIP) {
				new_h_q_f = (float) SSM_H_CLIP;
			}
			if (new_h_q_f < -(float) SSM_H_CLIP) {
				new_h_q_f = -(float) SSM_H_CLIP;
			}
			ssm_h_t new_h_q = (ssm_h_t) new_h_q_f;
			h[c][n] = new_h_q;

			y_scan_acc += (int32_t) new_h_q * (int32_t) w->C_q[n];
			/* Max |y_scan_acc|: SSM_H_CLIP (up to 32767 for int16) * 127 *
			 * SSM_D_STATE. At d_state = 8 that is ~3.33e7 -- safe in
			 * int32. Re-check this bound if d_state grows past ~500. */
		}

		float y_scan_real = (float) y_scan_acc * w->s_h * w->s_C;
		int8_t y_scan_q = ssm_requantize(y_scan_real, w->s_y_scan);

		int8_t silu_z_q = ssm_apply_lut(w->lut_silu_z, z_q[c]);
		float y_pre_gate_real = (float) y_scan_q * w->s_y_scan
				+ w->D[c] * (float) u_q[c] * w->s_u_post_conv_silu;
		float y_gated_real = y_pre_gate_real * ((float) silu_z_q * w->s_silu_z);
		y_gated_q[c] = ssm_requantize(y_gated_real, w->s_y_gated);
	}

	ssm_qlinear_real(y_gated_q, w->out_proj_w_q, SSM_D_MODEL, SSM_D_INNER,
			w->s_y_gated, w->out_proj_w_scale, NULL, block_out_real);
	ssm_quantize_arr(block_out_real, w->s_block_output, SSM_D_MODEL, block_out_q);
}

void SSMBackbone_Reset(SSMBackbone_State *state) {
	memset(state, 0, sizeof(*state));
}

void SSMBackbone_ProcessFrame(SSMBackbone_State *state, const float *frame_in, float *final_norm_out) {
	float x_real[SSM_D_MODEL];
	memcpy(x_real, frame_in, sizeof(x_real));

	for (int layer = 0; layer < SSM_N_LAYERS; layer++) {
		const ssm_true_int8_layer_t *w = &ssm_layers[layer];

		float x_norm_real[SSM_D_MODEL];
		ssm_rmsnorm(x_real, w->norm_w, x_norm_real);
		int8_t x_norm_q[SSM_D_MODEL];
		ssm_quantize_arr(x_norm_real, w->s_norm_out, SSM_D_MODEL, x_norm_q);

		int8_t block_out_q[SSM_D_MODEL];
		ssm_block_step_true_int8(w, state->h[layer], state->conv_hist_q[layer],
				x_norm_q, block_out_q);

		for (int i = 0; i < SSM_D_MODEL; i++) {
			x_real[i] += (float) block_out_q[i] * w->s_block_output;
		}
	}

	float normed_real[SSM_D_MODEL];
	ssm_rmsnorm(x_real, ssm_final_norm_w, normed_real);
	int8_t normed_q[SSM_D_MODEL];
	ssm_quantize_arr(normed_real, ssm_final_norm_scale, SSM_D_MODEL, normed_q);

	for (int i = 0; i < SSM_D_MODEL; i++) {
		float normed_roundtrip = (float) normed_q[i] * ssm_final_norm_scale;
		state->pooled_sum[i] += normed_roundtrip;
		if (final_norm_out != NULL) {
			final_norm_out[i] = normed_roundtrip;
		}
	}
	state->frame_count++;
}

void SSMBackbone_GetPooled(const SSMBackbone_State *state, float *pooled_out) {
	float inv_n = 1.0f / (float) state->frame_count;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		float pooled_real = state->pooled_sum[i] * inv_n;
		int8_t pooled_q = ssm_requantize(pooled_real, ssm_final_norm_scale);
		pooled_out[i] = (float) pooled_q * ssm_final_norm_scale;
	}
}

void SSMBackbone_NormalizeFrame(const float *raw, float *normalized) {
	for (int i = 0; i < SSM_D_MODEL; i++) {
		normalized[i] = (raw[i] - ssm_norm_mean[i]) / ssm_norm_std[i];
	}
}