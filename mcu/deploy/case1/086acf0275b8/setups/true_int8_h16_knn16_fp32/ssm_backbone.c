#include "ssm_backbone.h"

#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------------
 * Quantization primitives -- mirror true_int8_sim.py's quantize_int8 /
 * quantized_linear / apply_lut exactly. Every matmul below genuinely
 * accumulates in int32 and requantizes to int8 (or, for h, ssm_h_t) --
 * this is NOT the per-access dequantize-to-float pattern the fake-quant
 * backbones use.
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

/* Row-major int8 matvec: x_q (in_dim,) int8, w_q (out_dim, in_dim) int8 ->
 * out_real (out_dim,) real units. int32 accumulation is exact and safe
 * here (max |acc| <= 127*127*in_dim; for this model's largest in_dim,
 * SSM_D_MODEL/SSM_D_INNER = 64, that's ~1.03e6, nowhere near int32's
 * ~2.1e9 range) -- matches "genuinely accumulate in int32" from
 * true_int8_sim.py's module docstring (that file uses int64 for extra
 * safety margin; int32 is what a real kernel would use, and is what this
 * model's dimensions actually require). */
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
 *     function -- same "no clean int8 form" exception as an int8 FFT)
 *   - biases, added in real units at each op's rescale point
 *   - A itself, a static precomputed real array (never re-quantized)
 * Every other value that crosses a function boundary here is int8 (or,
 * for h, ssm_h_t), with its scale tracked separately -- mirrors
 * true_int8_sim.py's quantized_block_step line for line. */
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

	/* in_proj, split into u_raw and z. Both requantized against
	 * s_conv_out / s_z_gate respectively -- NOT a shared scale; matches
	 * quantized_linear's two separate calls in the Python reference. */
	ssm_qlinear_real(x_norm_q, w->in_proj_u_w_q, SSM_D_INNER, SSM_D_MODEL,
			w->s_norm_out, w->in_proj_w_scale, NULL, u_raw_real);
	ssm_qlinear_real(x_norm_q, w->in_proj_z_w_q, SSM_D_INNER, SSM_D_MODEL,
			w->s_norm_out, w->in_proj_w_scale, NULL, z_real);
	for (int c = 0; c < SSM_D_INNER; c++) {
		u_raw_q[c] = ssm_requantize(u_raw_real[c], w->s_conv_out);
		z_q[c] = ssm_requantize(z_real[c], w->s_z_gate);
	}

	/* Depthwise causal conv: tap 0 = oldest (t-3) .. tap D_CONV-1 =
	 * current (u_raw_q, this frame). conv_hist stores PRE-conv u_raw_q
	 * values, oldest-first -- same design as the fake-quant backbone and
	 * matches quantized_block_step's new_conv_hist_q[:, -1] = u_raw_q
	 * (not the post-conv-SiLU u_q). All taps share s_conv_out's scale
	 * domain, since conv_hist holds nothing but past u_raw_q values. */
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

	/* x_proj on post-conv u -> delta_low / B / C, real units first, then
	 * split and requantized to each field's own calibrated scale. No
	 * bias -- x_proj has none, same as every fake-quant scheme. */
	float x_dbl_real[SSM_DT_RANK + 2 * SSM_D_STATE];
	ssm_qlinear_real(u_q, w->x_proj_w_q, SSM_DT_RANK + 2 * SSM_D_STATE, SSM_D_INNER,
			w->s_u_post_conv_silu, w->x_proj_w_scale, NULL, x_dbl_real);

	int8_t delta_low_q[SSM_DT_RANK];
	int8_t B_q[SSM_D_STATE];
	int8_t C_q[SSM_D_STATE];
	for (int i = 0; i < SSM_DT_RANK; i++) {
		delta_low_q[i] = ssm_requantize(x_dbl_real[i], w->s_x_proj_delta_low_out);
	}
	for (int n = 0; n < SSM_D_STATE; n++) {
		B_q[n] = ssm_requantize(x_dbl_real[SSM_DT_RANK + n], w->s_B);
		C_q[n] = ssm_requantize(x_dbl_real[SSM_DT_RANK + SSM_D_STATE + n], w->s_C);
	}

	/* dt_proj (with bias) + softplus via LUT. */
	float delta_raw_real[SSM_D_INNER];
	ssm_qlinear_real(delta_low_q, w->dt_proj_w_q, SSM_D_INNER, SSM_DT_RANK,
			w->s_x_proj_delta_low_out, w->dt_proj_w_scale, w->dt_proj_b, delta_raw_real);
	int8_t delta_q[SSM_D_INNER];
	for (int c = 0; c < SSM_D_INNER; c++) {
		int8_t delta_raw_q = ssm_requantize(delta_raw_real[c], w->s_dt_proj_out);
		delta_q[c] = ssm_apply_lut(w->lut_softplus, delta_raw_q);
	}

	/* Euler discretize + recurrence + output, fused per channel. A_bar is
	 * computed in real units (A is a static real array, never itself
	 * re-quantized at runtime) then requantized; B_bar is computed from
	 * the RAW int8 values of delta_q/B_q multiplied as integers first,
	 * then scaled -- genuine int32 arithmetic, matching
	 * quantized_block_step's B_bar_real line exactly (not
	 * delta_real*B_real, which would be a different, less faithful,
	 * computation). */
	float block_out_real[SSM_D_MODEL];
	int8_t y_gated_q[SSM_D_INNER];

	for (int c = 0; c < SSM_D_INNER; c++) {
		float delta_real = (float) delta_q[c] * w->s_delta;

		int32_t y_scan_acc = 0;
		for (int n = 0; n < SSM_D_STATE; n++) {
			float deltaA_real = delta_real * w->A[c * SSM_D_STATE + n];
			if (deltaA_real < -1.9f) {
				deltaA_real = -1.9f;
			}
			float A_bar_real = 1.0f + deltaA_real;
			int8_t A_bar_q = ssm_requantize(A_bar_real, w->s_A_bar);

			float B_bar_real = (float) ((int32_t) delta_q[c] * (int32_t) B_q[n])
					* w->s_delta * w->s_B;
			int8_t B_bar_q = ssm_requantize(B_bar_real, w->s_B_bar);

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

			y_scan_acc += (int32_t) new_h_q * (int32_t) C_q[n];
			/* Max |y_scan_acc| over SSM_D_STATE=16 terms: SSM_H_CLIP (up
			 * to 32767 for int16) * 127 * 16 ~= 6.66e7 -- safe in int32. */
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
	/* Matches quantized_forward's final step exactly: the mean of already-
	 * quantized per-frame values is itself re-quantized before being
	 * returned. Without this, the "pooled embedding" would silently be a
	 * real-valued average -- not what an int8-embedding device produces. */
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