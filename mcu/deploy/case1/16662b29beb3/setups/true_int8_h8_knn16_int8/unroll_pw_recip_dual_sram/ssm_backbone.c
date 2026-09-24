#include "ssm_backbone.h"

#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------------
 * SRAM placement (plan 540, L1.4). In the arduino-pico linker script the
 * .data output section includes *(.time_critical*), and .data is copied
 * from flash to RAM at boot, so a section attribute is enough -- no
 * linker change. Same section name the sweep's --sram patch used, so
 * results stay comparable with the earlier q15-sram run.
 * --------------------------------------------------------------------- */
#define SSM_RAM_FUNC __attribute__((section(".time_critical.ssm_code")))

/* ---------------------------------------------------------------------
 * Quantization primitives -- mirror true_int8_sim.py's quantize_int8 /
 * quantized_linear / apply_lut exactly. Every matmul below genuinely
 * accumulates in int32 and requantizes to int8 (or, for h, ssm_h_t) --
 * this is NOT the per-access dequantize-to-float pattern the fake-quant
 * backbones use.
 * --------------------------------------------------------------------- */

SSM_RAM_FUNC static inline int8_t ssm_requantize(float real, float scale) {
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

/* Row-major int8 matvec: x_q (in_dim,) int8, w_q (out_dim, in_dim) int8 ->
 * out_real (out_dim,) real units. int32 accumulation is exact and safe
 * here (max |acc| <= 127*127*in_dim; for this model's largest in_dim,
 * SSM_D_MODEL/SSM_D_INNER = 64, that's ~1.03e6, nowhere near int32's
 * ~2.1e9 range) -- matches "genuinely accumulate in int32" from
 * true_int8_sim.py's module docstring (that file uses int64 for extra
 * safety margin; int32 is what a real kernel would use, and is what this
 * model's dimensions actually require).
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

/*
 * Ordering: volatile orders volatile accesses only relative to each
 * other. The context struct is a normal local on core0's stack, and at
 * -Os GCC moved ctx stores (including ctx->w) past the g_job_ready
 * store -- confirmed in the disassembly. Core1 then ran with a stale
 * weights pointer, faulted, and never set g_job_done. The fences below
 * are compiler barriers plus a DMB, and they are required.
 *
 * Sleep: both cores wait with WFE instead of busy-polling, so a waiting
 * core stops competing for the bus and the XIP cache. The poster wakes
 * the other core with SEV. No wakeup can be lost: if the SEV lands
 * between a core's flag check and its WFE, the event register stays
 * set and the WFE returns at once. Other wake sources (interrupts,
 * SEVs from pico-sdk spinlock and mutex code) only cause a re-check of
 * the flag. */
typedef void (*ssm_core1_work_fn)(void *ctx);

static volatile ssm_core1_work_fn g_work_fn = NULL;
static void * volatile g_work_ctx = NULL;
static volatile int g_job_ready = 0;
static volatile int g_job_done = 0;

/* Own names instead of the pico-sdk __wfe()/__sev(), so this file does
 * not depend on which SDK headers arrive transitively. The "memory"
 * clobber makes each one a compiler barrier too. DSB before SEV makes
 * sure the flag store has completed before the other core wakes. */
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
	/* Release: all stores to *ctx and its input arrays complete first. */
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	g_job_ready = 1;
	ssm_sev();
}

SSM_RAM_FUNC static void ssm_wait_for_core1(void) {
	while (!g_job_done) {
		ssm_wfe();
	}
	/* Acquire: no read of core1's outputs moves above the wait. */
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
}

/* Called from loop1() on core1. Blocks (asleep in WFE) until core0
 * posts a job, runs it, signals done, and returns -- loop1() then calls
 * it again for the next job. */
SSM_RAM_FUNC void ssm_core1_loop_step(void) {
	while (!g_job_ready) {
		ssm_wfe();
	}
	g_job_ready = 0;
	/* Acquire: read the job only after seeing the flag. */
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	ssm_core1_work_fn fn = g_work_fn;
	void *ctx = g_work_ctx;
	fn(ctx);
	/* Release: all output stores complete before core0 sees done. */
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	g_job_done = 1;
	ssm_sev();
}

/* --- x_proj / out_proj: whole-vector ops, split by output row, exactly
 * as before -- each needs its full input vector, so these stay their
 * own dispatch. --- */

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
 * channel. Each channel's conv only touches its own conv_hist_q[c] and
 * u_raw_q[c] -- no cross-channel dependency. --- */

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

/* --- Phase 3: dt_proj (folded in -- in_dim = SSM_DT_RANK is too small
 * to dispatch on its own) + the Euler-discretize/scan/y_gated
 * recurrence, split by channel. h[c] and conv_hist_q[c] carry no
 * cross-channel dependency, so this is the biggest chunk of work in the
 * function and the cleanest to parallelize. --- */

typedef struct {
	const ssm_true_int8_layer_t *w;
	ssm_h_t (*h)[SSM_D_STATE];
	const int8_t *delta_low_q;
	const int8_t *B_q;
	const int8_t *C_q;
	const int8_t *u_q;
	const int8_t *z_q;
	int c_start;
	int c_end;
	float inv_s_dt_proj_out;
	float inv_s_A_bar;
	float inv_s_B_bar;
	float inv_s_h;
	float inv_s_y_scan;
	float inv_s_y_gated;
	int8_t *y_gated_q;
} ssm_phase3_ctx_t;

SSM_RAM_FUNC static void ssm_phase3_work(void *ctx_v) {
	ssm_phase3_ctx_t *ctx = (ssm_phase3_ctx_t *) ctx_v;
	const ssm_true_int8_layer_t *w = ctx->w;

	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		/* dt_proj, row c. */
		const int8_t *dt_row = &w->dt_proj_w_q[c * SSM_DT_RANK];
		int32_t dt_acc = 0;
		for (int i = 0; i < SSM_DT_RANK; i++) {
			dt_acc += (int32_t) dt_row[i] * (int32_t) ctx->delta_low_q[i];
		}
		float delta_raw_real = (float) dt_acc * w->s_x_proj_delta_low_out * w->dt_proj_w_scale
				+ w->dt_proj_b[c];
		int8_t delta_raw_q = ssm_requantize_r(delta_raw_real, ctx->inv_s_dt_proj_out);
		int8_t delta_q_c = ssm_apply_lut(w->lut_softplus, delta_raw_q);

		float delta_real = (float) delta_q_c * w->s_delta;

		int32_t y_scan_acc = 0;
		for (int n = 0; n < SSM_D_STATE; n++) {
			float deltaA_real = delta_real * w->A[c * SSM_D_STATE + n];
			if (deltaA_real < -1.9f) {
				deltaA_real = -1.9f;
			}
			float A_bar_real = 1.0f + deltaA_real;
			int8_t A_bar_q = ssm_requantize_r(A_bar_real, ctx->inv_s_A_bar);

			float B_bar_real = (float) ((int32_t) delta_q_c * (int32_t) ctx->B_q[n])
					* w->s_delta * w->s_B;
			int8_t B_bar_q = ssm_requantize_r(B_bar_real, ctx->inv_s_B_bar);

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

			y_scan_acc += (int32_t) new_h_q * (int32_t) ctx->C_q[n];
			/* Max |y_scan_acc| over SSM_D_STATE=16 terms: SSM_H_CLIP (up
			 * to 32767 for int16) * 127 * 16 ~= 6.66e7 -- safe in int32. */
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
 * true_int8_sim.py's quantized_block_step line for line.
 */
SSM_RAM_FUNC static void ssm_block_step_true_int8(
		const ssm_true_int8_layer_t *w,
		ssm_h_t h[SSM_D_INNER][SSM_D_STATE],
		int8_t conv_hist_q[SSM_D_INNER][SSM_D_CONV - 1],
		const int8_t *x_norm_q,
		int8_t *block_out_q) {

	float inv_s_conv_out = 1.0f / w->s_conv_out;
	float inv_s_z_gate = 1.0f / w->s_z_gate;
	float inv_s_x_proj_delta_low_out = 1.0f / w->s_x_proj_delta_low_out;
	float inv_s_B = 1.0f / w->s_B;
	float inv_s_C = 1.0f / w->s_C;
	float inv_s_dt_proj_out = 1.0f / w->s_dt_proj_out;
	float inv_s_A_bar = 1.0f / w->s_A_bar;
	float inv_s_B_bar = 1.0f / w->s_B_bar;
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

	/* x_proj on post-conv u -> delta_low / B / C, real units first, then
	 * split and requantized to each field's own calibrated scale. No
	 * bias -- x_proj has none, same as every fake-quant scheme. */
	float x_dbl_real[SSM_DT_RANK + 2 * SSM_D_STATE];
	ssm_qlinear_real_parallel(u_q, w->x_proj_w_q, SSM_DT_RANK + 2 * SSM_D_STATE, SSM_D_INNER,
			w->s_u_post_conv_silu, w->x_proj_w_scale, NULL, x_dbl_real);

	int8_t delta_low_q[SSM_DT_RANK];
	int8_t B_q[SSM_D_STATE];
	int8_t C_q[SSM_D_STATE];
	for (int i = 0; i < SSM_DT_RANK; i++) {
		delta_low_q[i] = ssm_requantize_r(x_dbl_real[i], inv_s_x_proj_delta_low_out);
	}
	for (int n = 0; n < SSM_D_STATE; n++) {
		B_q[n] = ssm_requantize_r(x_dbl_real[SSM_DT_RANK + n], inv_s_B);
		C_q[n] = ssm_requantize_r(x_dbl_real[SSM_DT_RANK + SSM_D_STATE + n], inv_s_C);
	}

	int8_t y_gated_q[SSM_D_INNER];
	{
		int half = SSM_D_INNER / 2;
		ssm_phase3_ctx_t ctx1 = {
			w, h, delta_low_q, B_q, C_q, u_q, z_q, half, SSM_D_INNER,
			inv_s_dt_proj_out, inv_s_A_bar, inv_s_B_bar, inv_s_h, inv_s_y_scan, inv_s_y_gated,
			y_gated_q,
		};
		ssm_dispatch_to_core1(ssm_phase3_work, &ctx1);

		ssm_phase3_ctx_t ctx0 = {
			w, h, delta_low_q, B_q, C_q, u_q, z_q, 0, half,
			inv_s_dt_proj_out, inv_s_A_bar, inv_s_B_bar, inv_s_h, inv_s_y_scan, inv_s_y_gated,
			y_gated_q,
		};
		ssm_phase3_work(&ctx0);

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
	/* Matches quantized_forward's final step exactly: the mean of already-
	 * quantized per-frame values is itself re-quantized before being
	 * returned. Without this, the "pooled embedding" would silently be a
	 * real-valued average -- not what an int8-embedding device produces. */
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