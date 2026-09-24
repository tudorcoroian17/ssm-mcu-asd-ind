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
 * Quantization primitives, the dual-core hand-off, and SRAM placement
 * are unchanged from the selective true-int8 source, deliberately, so
 * the two can still be diffed on everything that isn't scheme-specific.
 * --------------------------------------------------------------------- */
#define SSM_RAM_FUNC __attribute__((section(".time_critical.ssm_code")))

static inline int8_t ssm_requantize(float real, float scale) {
	float q = roundf(real / scale);
	if (q > 127.0f) q = 127.0f;
	if (q < -127.0f) q = -127.0f;
	return (int8_t) q;
}

SSM_RAM_FUNC static inline int8_t ssm_requantize_r(float real, float inv_scale) {
	float q = roundf(real * inv_scale);
	if (q > 127.0f) q = 127.0f;
	if (q < -127.0f) q = -127.0f;
	return (int8_t) q;
}

SSM_RAM_FUNC static void ssm_quantize_arr(const float *real, float scale, int n, int8_t *out_q) {
	float inv_scale = 1.0f / scale;
	for (int i = 0; i < n; i++) {
		out_q[i] = ssm_requantize_r(real[i], inv_scale);
	}
}

SSM_RAM_FUNC static inline int8_t ssm_apply_lut(const int8_t *lut, int8_t x_q) {
	return lut[(int) x_q + 128];
}

/* 	The euler step from ssm_block.py's discretize(). Used only by the qab
 * 	macro expansion; unused (and harmless) in a qabar build. */
SSM_RAM_FUNC static inline float ssm_euler_abar(float deltaA) {
	if (deltaA < -1.9f) {
		deltaA = -1.9f;
	}
	return 1.0f + deltaA;
}

/* Row-major int8 matvec: x_q (in_dim,) int8, w_q (out_dim, in_dim) int8 ->
 * out_real (out_dim,) real units. int32 accumulation is exact and safe
 * here: max |acc| <= 127*127*in_dim, and the largest in_dim in this model
 * is SSM_D_INNER, so for d_inner = 128 that is ~2.06e6, well inside
 * int32's ~2.1e9 range.
 */
SSM_RAM_FUNC static void ssm_qlinear_real_range(const int8_t *x_q, const int8_t *w_q,
		int out_start, int out_end, int in_dim, float x_scale, float w_scale,
		const float *bias, float *out_real) {
	for (int o = out_start; o < out_end; o++) {
		const int8_t *w_row = &w_q[o * in_dim];
		const int8_t *x_p = x_q;
		int32_t acc = 0;
		int n = in_dim;
		while (n >= 4) {
			acc += (int32_t) w_row[0] * (int32_t) x_p[0];
			acc += (int32_t) w_row[1] * (int32_t) x_p[1];
			acc += (int32_t) w_row[2] * (int32_t) x_p[2];
			acc += (int32_t) w_row[3] * (int32_t) x_p[3];
			w_row += 4;
			x_p += 4;
			n -= 4;
		}
		while (n > 0) {
			acc += (int32_t) (*w_row++) * (int32_t) (*x_p++);
			n--;
		}
		float real = (float) acc * x_scale * w_scale;
		if (bias != NULL) {
			real += bias[o];
		}
		out_real[o] = real;
	}
}

typedef void (*ssm_core1_work_fn)(void *ctx);

static volatile ssm_core1_work_fn g_work_fn = NULL;
static void * volatile g_work_ctx = NULL;
static volatile int g_job_ready = 0;
static volatile int g_job_done = 0;

SSM_RAM_FUNC static inline void ssm_wfe(void) {
	__asm volatile ("wfe" ::: "memory");
}

SSM_RAM_FUNC static inline void ssm_sev(void) {
	__asm volatile ("dsb\n\tsev" ::: "memory");
}

SSM_RAM_FUNC static void ssm_dispatch_to_core1(ssm_core1_work_fn fn, void *ctx) {
	g_work_fn = fn;
	g_work_ctx = ctx;
	g_job_done = 0;
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	g_job_ready = 1;
	ssm_sev();
}

SSM_RAM_FUNC static void ssm_wait_for_core1(void) {
	while (!g_job_done) {
		ssm_wfe();
	}
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
}

SSM_RAM_FUNC void ssm_core1_loop_step(void) {
	while (!g_job_ready) {
		ssm_wfe();
	}
	g_job_ready = 0;
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	ssm_core1_work_fn fn = g_work_fn;
	void *ctx = g_work_ctx;
	fn(ctx);
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	g_job_done = 1;
	ssm_sev();
}

/* --- out_proj: whole-vector op, split by output row. --- */

typedef struct {
	const int8_t *x_q;
	const int8_t *w_q;
	int out_start;
	int out_end;
	int in_dim;
	float x_scale;
	float w_scale;
	const float *bias;
	float *out_real;
} ssm_qlinear_ctx_t;

SSM_RAM_FUNC static void ssm_qlinear_work(void *ctx_v) {
	ssm_qlinear_ctx_t *ctx = (ssm_qlinear_ctx_t *) ctx_v;
	ssm_qlinear_real_range(ctx->x_q, ctx->w_q, ctx->out_start, ctx->out_end, ctx->in_dim,
			ctx->x_scale, ctx->w_scale, ctx->bias, ctx->out_real);
}

SSM_RAM_FUNC static void ssm_qlinear_real_parallel(const int8_t *x_q, const int8_t *w_q,
		int out_dim, int in_dim, float x_scale, float w_scale,
		const float *bias, float *out_real) {
	int half = out_dim / 2;
	ssm_qlinear_ctx_t ctx1 = { x_q, w_q, half, out_dim, in_dim, x_scale, w_scale, bias, out_real };
	ssm_dispatch_to_core1(ssm_qlinear_work, &ctx1);

	ssm_qlinear_real_range(x_q, w_q, 0, half, in_dim, x_scale, w_scale, bias, out_real);

	ssm_wait_for_core1();
}

/* --- Phase 1: in_proj_u, in_proj_z, and the conv step, split by
 * channel -- byte-identical to selective's Phase 1. --- */

typedef struct {
	const ssm_true_int8_layer_t *w;
	const int8_t *x_norm_q;
	int8_t (*conv_hist_q)[SSM_D_CONV - 1];
	int c_start;
	int c_end;
	float inv_s_conv_out;
	float inv_s_z_gate;
	float *u_raw_real;
	float *z_real;
	int8_t *u_raw_q;
	int8_t *z_q;
	int8_t *u_q;
} ssm_phase1_ctx_t;

SSM_RAM_FUNC static void ssm_phase1_work(void *ctx_v) {
	ssm_phase1_ctx_t *ctx = (ssm_phase1_ctx_t *) ctx_v;
	const ssm_true_int8_layer_t *w = ctx->w;

	ssm_qlinear_real_range(ctx->x_norm_q, w->in_proj_u_w_q, ctx->c_start, ctx->c_end,
			SSM_D_MODEL, w->s_norm_out, w->in_proj_w_scale, NULL, ctx->u_raw_real);
	ssm_qlinear_real_range(ctx->x_norm_q, w->in_proj_z_w_q, ctx->c_start, ctx->c_end,
			SSM_D_MODEL, w->s_norm_out, w->in_proj_w_scale, NULL, ctx->z_real);

	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		ctx->u_raw_q[c] = ssm_requantize_r(ctx->u_raw_real[c], ctx->inv_s_conv_out);
		ctx->z_q[c] = ssm_requantize_r(ctx->z_real[c], ctx->inv_s_z_gate);
	}

	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		const int8_t *w_row = &w->conv_w_q[c * SSM_D_CONV];
		int32_t acc = 0;
		for (int k = 0; k < SSM_D_CONV - 1; k++) {
			acc += (int32_t) w_row[k] * (int32_t) ctx->conv_hist_q[c][k];
		}
		acc += (int32_t) w_row[SSM_D_CONV - 1] * (int32_t) ctx->u_raw_q[c];

		float conv_real = (float) acc * w->s_conv_out * w->conv_w_scale + w->conv_b[c];
		int8_t conv_out_q = ssm_requantize_r(conv_real, ctx->inv_s_conv_out);
		ctx->u_q[c] = ssm_apply_lut(w->lut_silu_conv, conv_out_q);

		for (int k = 0; k < SSM_D_CONV - 2; k++) {
			ctx->conv_hist_q[c][k] = ctx->conv_hist_q[c][k + 1];
		}
		ctx->conv_hist_q[c][SSM_D_CONV - 2] = ctx->u_raw_q[c];
	}
}

/* --- Phase 2: recurrence + y_gated, split by channel. h[c] carries no
 * cross-channel dependency. No dt_proj, no runtime discretize -- A_bar_q
 * and B_bar_q come straight from the SSM_TI_* macros. --- */

typedef struct {
	const ssm_true_int8_layer_t *w;
	ssm_h_t (*h)[SSM_D_STATE];
	const int8_t *u_q;
	const int8_t *z_q;
	int c_start;
	int c_end;
	float inv_s_h;
	float inv_s_y_scan;
	float inv_s_y_gated;
	int8_t *y_gated_q;
} ssm_phase2_ctx_t;

SSM_RAM_FUNC static void ssm_phase2_work(void *ctx_v) {
	ssm_phase2_ctx_t *ctx = (ssm_phase2_ctx_t *) ctx_v;
	const ssm_true_int8_layer_t *w = ctx->w;

	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		const float delta_c = SSM_TI_DELTA_REAL(w, c);

		int32_t y_scan_acc = 0;
		for (int n = 0; n < SSM_D_STATE; n++) {
			int8_t A_bar_q = SSM_TI_A_BAR_Q(w, c, n, delta_c);
			int8_t B_bar_q = SSM_TI_B_BAR_Q(w, c, n, delta_c);

			float Ah_real = (float) ((int32_t) A_bar_q * (int32_t) ctx->h[c][n])
					* w->s_A_bar * w->s_h;
			float Bu_real = (float) ((int32_t) B_bar_q * (int32_t) ctx->u_q[c])
					* w->s_B_bar * w->s_u_post_conv_silu;
			float new_h_real = Ah_real + Bu_real;

			float new_h_q_f = roundf(new_h_real * ctx->inv_s_h);
			if (new_h_q_f > (float) SSM_H_CLIP) {
				new_h_q_f = (float) SSM_H_CLIP;
			}
			if (new_h_q_f < -(float) SSM_H_CLIP) {
				new_h_q_f = -(float) SSM_H_CLIP;
			}
			ssm_h_t new_h_q = (ssm_h_t) new_h_q_f;
			ctx->h[c][n] = new_h_q;

			y_scan_acc += (int32_t) new_h_q * (int32_t) w->C_q[n];
			/* Max |y_scan_acc|: SSM_H_CLIP (up to 32767 for int16) * 127 *
			 * SSM_D_STATE. At d_state = 8 that is ~3.33e7 -- safe in
			 * int32. Re-check this bound if d_state grows past ~500. */
		}

		float y_scan_real = (float) y_scan_acc * w->s_h * w->s_C;
		int8_t y_scan_q = ssm_requantize_r(y_scan_real, ctx->inv_s_y_scan);

		int8_t silu_z_q = ssm_apply_lut(w->lut_silu_z, ctx->z_q[c]);
		float y_pre_gate_real = (float) y_scan_q * w->s_y_scan
				+ w->D[c] * (float) ctx->u_q[c] * w->s_u_post_conv_silu;
		float y_gated_real = y_pre_gate_real * ((float) silu_z_q * w->s_silu_z);
		ctx->y_gated_q[c] = ssm_requantize_r(y_gated_real, ctx->inv_s_y_gated);
	}
}

SSM_RAM_FUNC static void ssm_rmsnorm(const float *x, const float *weight, float *out) {
	float ss = 0.0f;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		ss += x[i] * x[i];
	}
	float inv_rms = 1.0f / sqrtf(ss * (1.0f / (float) SSM_D_MODEL) + 1e-5f);
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
SSM_RAM_FUNC static void ssm_block_step_true_int8(
		const ssm_true_int8_layer_t *w,
		ssm_h_t h[SSM_D_INNER][SSM_D_STATE],
		int8_t conv_hist_q[SSM_D_INNER][SSM_D_CONV - 1],
		const int8_t *x_norm_q,
		int8_t *block_out_q) {

	float inv_s_conv_out = 1.0f / w->s_conv_out;
	float inv_s_z_gate = 1.0f / w->s_z_gate;
	float inv_s_h = 1.0f / w->s_h;
	float inv_s_y_scan = 1.0f / w->s_y_scan;
	float inv_s_y_gated = 1.0f / w->s_y_gated;

	float u_raw_real[SSM_D_INNER];
	float z_real[SSM_D_INNER];
	int8_t u_raw_q[SSM_D_INNER];
	int8_t z_q[SSM_D_INNER];
	int8_t u_q[SSM_D_INNER];

	{
		int half = SSM_D_INNER / 2;
		ssm_phase1_ctx_t ctx1 = {
			w, x_norm_q, conv_hist_q, half, SSM_D_INNER,
			inv_s_conv_out, inv_s_z_gate,
			u_raw_real, z_real, u_raw_q, z_q, u_q,
		};
		ssm_dispatch_to_core1(ssm_phase1_work, &ctx1);

		ssm_phase1_ctx_t ctx0 = {
			w, x_norm_q, conv_hist_q, 0, half,
			inv_s_conv_out, inv_s_z_gate,
			u_raw_real, z_real, u_raw_q, z_q, u_q,
		};
		ssm_phase1_work(&ctx0);

		ssm_wait_for_core1();
	}

	int8_t y_gated_q[SSM_D_INNER];
	{
		int half = SSM_D_INNER / 2;
		ssm_phase2_ctx_t ctx1 = {
			w, h, u_q, z_q, half, SSM_D_INNER,
			inv_s_h, inv_s_y_scan, inv_s_y_gated,
			y_gated_q,
		};
		ssm_dispatch_to_core1(ssm_phase2_work, &ctx1);

		ssm_phase2_ctx_t ctx0 = {
			w, h, u_q, z_q, 0, half,
			inv_s_h, inv_s_y_scan, inv_s_y_gated,
			y_gated_q,
		};
		ssm_phase2_work(&ctx0);

		ssm_wait_for_core1();
	}

	float block_out_real[SSM_D_MODEL];
	ssm_qlinear_real_parallel(y_gated_q, w->out_proj_w_q, SSM_D_MODEL, SSM_D_INNER,
			w->s_y_gated, w->out_proj_w_scale, NULL, block_out_real);
	ssm_quantize_arr(block_out_real, w->s_block_output, SSM_D_MODEL, block_out_q);
}

SSM_RAM_FUNC void SSMBackbone_Reset(SSMBackbone_State *state) {
	memset(state, 0, sizeof(*state));
}

SSM_RAM_FUNC void SSMBackbone_ProcessFrame(SSMBackbone_State *state, const float *frame_in, float *final_norm_out) {
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

SSM_RAM_FUNC void SSMBackbone_GetPooled(const SSMBackbone_State *state, float *pooled_out) {
	float inv_n = 1.0f / (float) state->frame_count;
	float inv_s_final = 1.0f / ssm_final_norm_scale;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		float pooled_real = state->pooled_sum[i] * inv_n;
		int8_t pooled_q = ssm_requantize_r(pooled_real, inv_s_final);
		pooled_out[i] = (float) pooled_q * ssm_final_norm_scale;
	}
}

SSM_RAM_FUNC void SSMBackbone_NormalizeFrame(const float *raw, float *normalized) {
	for (int i = 0; i < SSM_D_MODEL; i++) {
		normalized[i] = (raw[i] - ssm_norm_mean[i]) / ssm_norm_std[i];
	}
}