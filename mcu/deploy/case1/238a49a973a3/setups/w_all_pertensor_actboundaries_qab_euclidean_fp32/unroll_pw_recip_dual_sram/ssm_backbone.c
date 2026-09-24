#include "ssm_backbone.h"

#include <math.h>
#include <string.h>

/* CLASSIC (selective=False) backbone, quantized-float schemes. Same
 * differences from the selective source as ssm_backbone_classic_src:
 *
 *   - No x_proj, no dt_proj, no split. delta, B and C are static
 *     parameters, so nothing is projected out of u per frame.
 *   - The recurrence coefficients reach this file through SSM_DELTA,
 *     SSM_A_BAR and SSM_B_BAR, defined in the GENERATED ssm_weights.h.
 *     Under qab they expand to the live discretize math (they call
 *     ssm_softplus and ssm_euler_abar, so both must be defined in this
 *     file). Under qabar they read baked constant arrays and discard
 *     delta_c. This file's text is identical either way.
 *   - SSM_DT_RANK does not exist in the classic headers. Do not use it
 *     here.
 *
 * ---------------------------------------------------------------------
 * SRAM placement. Same section-attribute mechanism as the true_int8
 * backbones -- see those files for the linker-script rationale and the
 * long-branch-veneer caveat for soft-float calls (roundf, expf, sqrtf,
 * log1pf all still live in flash even with this file's code in RAM).
 * --------------------------------------------------------------------- */
#define SSM_RAM_FUNC __attribute__((section(".time_critical.ssm_code")))

static inline float ssm_softplus(float x) {
	/* 	Matches torch.nn.functional.softplus's default threshold=20: linear
	 * 	for large x, where log1p(exp(x)) would lose precision anyway. */
	return (x > 20.0f) ? x : log1pf(expf(x));
}

static inline float ssm_silu(float x) {
	return x / (1.0f + expf(-x));
}

/* 	The euler step from ssm_block.py's discretize(): clamp delta*A at -1.9
 * 	before adding 1, so A_bar stays above -0.9 and the recurrence cannot
 * 	diverge. Used only by the qab macro expansion; unused (and harmless)
 * 	in a qabar build. */
static inline float ssm_euler_abar(float deltaA) {
	if (deltaA < -1.9f) {
		deltaA = -1.9f;
	}
	return 1.0f + deltaA;
}

/* ---------------------------------------------------------------------
 * Activation quantize-dequantize, reciprocal-hoisted.
 *
 * Claude comment: same scale-extraction trick as the selective quantized
 * file (ssm_backbone_quant_src) -- see that file for the full reasoning.
 * Short version: each SSM_QUANT_* hook is either (val) or
 * (ssm_quant_dequant((val), <scale>)). The scale fields exist only in
 * actboundaries headers, so this file gets them through the hook: for the
 * two extraction functions only, ssm_quant_dequant is a macro that
 * returns its second argument, and the hooks are called with a -1.0f
 * sentinel. A result <= 0 means "this boundary is not quantized".
 *
 * NOT bit-exact under actboundaries (roundf(v * inv_s) vs roundf(v / s),
 * at most one LSB). Bit-exact for every other setup.
 * --------------------------------------------------------------------- */
typedef struct {
	float s;   /* scale; <= 0 means this boundary is not quantized */
	float inv; /* 1.0f / s, computed once per block call */
} ssm_qscale_t;

typedef struct {
	ssm_qscale_t z;
	ssm_qscale_t u;
	ssm_qscale_t y_gated;
	ssm_qscale_t block_out;
} ssm_act_scales_t;

#define ssm_quant_dequant(v, s) (s)

SSM_RAM_FUNC static void ssm_get_act_scales(const ssm_block_weights_t *w, ssm_act_scales_t *out) {
	(void) w; /* unused when every hook is the (val) form */
	out->z.s = SSM_QUANT_Z(w, -1.0f);
	out->u.s = SSM_QUANT_U(w, -1.0f);
	out->y_gated.s = SSM_QUANT_Y_GATED(w, -1.0f);
	out->block_out.s = SSM_QUANT_BLOCK_OUT(w, -1.0f);
}

SSM_RAM_FUNC static float ssm_get_final_norm_scale(void) {
	return SSM_QUANT_FINAL_NORM(-1.0f);
}

#undef ssm_quant_dequant

SSM_RAM_FUNC static inline void ssm_qscale_finish(ssm_qscale_t *q) {
	q->inv = (q->s > 0.0f) ? (1.0f / q->s) : 0.0f;
}

/* Same clamp and rounding as the original ssm_quant_dequant, with the
 * division replaced by a multiply. No-op when the boundary is off. */
SSM_RAM_FUNC static inline float ssm_qdq_r(float v, const ssm_qscale_t *q) {
	if (q->s <= 0.0f) {
		return v;
	}
	float r = roundf(v * q->inv);
	if (r > 127.0f) r = 127.0f;
	if (r < -127.0f) r = -127.0f;
	return r * q->s;
}

SSM_RAM_FUNC static void ssm_rmsnorm(const float *x, const ssm_norm_weights_t *nw, float *out) {
	float ss = 0.0f;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		ss += x[i] * x[i];
	}
	float inv_rms = 1.0f / sqrtf(ss * (1.0f / (float) SSM_D_MODEL) + 1e-5f);
	for (int i = 0; i < SSM_D_MODEL; i++) {
		out[i] = x[i] * inv_rms * SSM_NORM_W(nw, i);
	}
}

/* Claude comment: dual-core hand-off, identical mechanism to the
 * true_int8 backbones -- see ssm_true_int8_src/ssm_backbone.c for the
 * full ordering rationale. */
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

/* ---------------------------------------------------------------------
 * Matvecs: one per weight matrix (in_proj, out_proj), because each
 * matrix has its own SSM_*_W macro.
 *
 * Claude comment: identical to the selective quantized file. Weights go
 * through the macro (pertensor scale[0] vs perchannel scale[r] stays
 * hidden, no out-of-bounds read); only the input vector is
 * pointer-walked. Unrolled 4x, one accumulator, same left-to-right order
 * as the original loop -> bit-exact. The tail loop is dead for the
 * current dims; it stays for safety.
 * --------------------------------------------------------------------- */

/* row_base = 0 for u rows, SSM_D_INNER for z rows (in_proj is one matrix
 * with 2 * SSM_D_INNER rows). out[o] is written directly. */
SSM_RAM_FUNC static void ssm_in_proj_range(const ssm_block_weights_t *w, int row_base,
		const float *x, int o_start, int o_end, float *out) {
	for (int o = o_start; o < o_end; o++) {
		const int r = row_base + o;
		const float *x_p = x;
		float acc = 0.0f;
		int i = 0;
		while (i + 4 <= SSM_D_MODEL) {
			acc += SSM_IN_PROJ_W(w, r, i) * x_p[0];
			acc += SSM_IN_PROJ_W(w, r, i + 1) * x_p[1];
			acc += SSM_IN_PROJ_W(w, r, i + 2) * x_p[2];
			acc += SSM_IN_PROJ_W(w, r, i + 3) * x_p[3];
			x_p += 4;
			i += 4;
		}
		while (i < SSM_D_MODEL) {
			acc += SSM_IN_PROJ_W(w, r, i) * (*x_p++);
			i++;
		}
		out[o] = acc;
	}
}

SSM_RAM_FUNC static void ssm_out_proj_range(const ssm_block_weights_t *w,
		const float *x, int o_start, int o_end, float *out) {
	for (int o = o_start; o < o_end; o++) {
		const float *x_p = x;
		float acc = 0.0f;
		int i = 0;
		while (i + 4 <= SSM_D_INNER) {
			acc += SSM_OUT_PROJ_W(w, o, i) * x_p[0];
			acc += SSM_OUT_PROJ_W(w, o, i + 1) * x_p[1];
			acc += SSM_OUT_PROJ_W(w, o, i + 2) * x_p[2];
			acc += SSM_OUT_PROJ_W(w, o, i + 3) * x_p[3];
			x_p += 4;
			i += 4;
		}
		while (i < SSM_D_INNER) {
			acc += SSM_OUT_PROJ_W(w, o, i) * (*x_p++);
			i++;
		}
		out[o] = acc;
	}
}

/* --- out_proj: whole-vector op, split by output row. --- */

typedef struct {
	const ssm_block_weights_t *w;
	const float *x;
	int o_start;
	int o_end;
	float *out;
} ssm_proj_ctx_t;

SSM_RAM_FUNC static void ssm_out_proj_work(void *ctx_v) {
	ssm_proj_ctx_t *ctx = (ssm_proj_ctx_t *) ctx_v;
	ssm_out_proj_range(ctx->w, ctx->x, ctx->o_start, ctx->o_end, ctx->out);
}

/* ---------------------------------------------------------------------
 * Channel phase: in_proj (u and z rows) + conv + recurrence + y_gated,
 * fused, split by channel.
 *
 * Claude comment: this is ONE phase, not two. In the classic branch,
 * channel c of the scan needs only u[c], z[c], h[c] and static weights --
 * there is no x_proj that mixes all channels of u before the scan. So a
 * core that computes u[c] and z[c] for its channel range can run the
 * scan for the same range with no barrier in between. Result: 2 round
 * trips per block (this phase, then out_proj) instead of 3.
 *
 * The z, u and y_gated hooks are element-wise, so applying them per
 * channel inside each core's range gives the same values as the
 * original "whole array, then quantize" order. u_raw, z and u are
 * per-core scratch here (only this core's range is ever read back), so
 * the caller passes one full-size set of arrays and each core uses its
 * own slice.
 * --------------------------------------------------------------------- */

typedef struct {
	const ssm_block_weights_t *w;
	const float *x_norm;
	float (*conv_hist)[SSM_D_CONV - 1];
	float (*h)[SSM_D_STATE];
	const ssm_act_scales_t *q;
	int c_start;
	int c_end;
	float *u_raw;
	float *z;
	float *u;
	float *y;
} ssm_channel_ctx_t;

SSM_RAM_FUNC static void ssm_channel_work(void *ctx_v) {
	ssm_channel_ctx_t *ctx = (ssm_channel_ctx_t *) ctx_v;
	const ssm_block_weights_t *w = ctx->w;

	/* in_proj: u rows and z rows for this channel range. */
	ssm_in_proj_range(w, 0, ctx->x_norm, ctx->c_start, ctx->c_end, ctx->u_raw);
	ssm_in_proj_range(w, SSM_D_INNER, ctx->x_norm, ctx->c_start, ctx->c_end, ctx->z);

	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		ctx->z[c] = ssm_qdq_r(ctx->z[c], &ctx->q->z);
	}

	/*	causal depthwise conv: kernel tap 0 = oldest (t-3) .. tap 3 = current
	 * 	(t), then SiLU. conv_hist holds t-3,t-2,t-1 oldest-first. */
	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		float acc = SSM_CONV_B(w, c);
		for (int k = 0; k < SSM_D_CONV - 1; k++) {
			acc += SSM_CONV_W(w, c, k) * ctx->conv_hist[c][k];
		}
		acc += SSM_CONV_W(w, c, SSM_D_CONV - 1) * ctx->u_raw[c];
		ctx->u[c] = ssm_silu(acc);

		for (int k = 0; k < SSM_D_CONV - 2; k++) {
			ctx->conv_hist[c][k] = ctx->conv_hist[c][k + 1];
		}
		ctx->conv_hist[c][SSM_D_CONV - 2] = ctx->u_raw[c];
	}
	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		ctx->u[c] = ssm_qdq_r(ctx->u[c], &ctx->q->u);
	}

	/* 	Recurrence + output, per channel. delta_c is hoisted out of the
	 * 	inner loop, same as the original: under qab it is
	 * 	softplus(dequantized dt[c]); under qabar it is 0.0f and both
	 * 	coefficient macros discard it via the comma operator.
	 *
	 * 	Claude comment: the scan loop is left as plain macro calls, not
	 * 	unrolled -- same known gap as every other backbone in this set. */
	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		const float delta_c = SSM_DELTA(w, c);
		const float u_c = ctx->u[c];
		float y_c = 0.0f;
		for (int n = 0; n < SSM_D_STATE; n++) {
			float A_bar = SSM_A_BAR(w, c, n, delta_c);
			float B_bar = SSM_B_BAR(w, c, n, delta_c);
			float new_h = A_bar * ctx->h[c][n] + B_bar * u_c;
			ctx->h[c][n] = new_h;
			y_c += new_h * SSM_C(w, n);
		}
		float y_val = (y_c + u_c * SSM_D_PARAM(w, c)) * ssm_silu(ctx->z[c]);
		ctx->y[c] = ssm_qdq_r(y_val, &ctx->q->y_gated);
	}
}

/* Claude comment: 2 dual-core round trips per block -- channel phase,
 * then out_proj. out_proj is the only step that needs every channel of
 * y, so it is the only real barrier. */
SSM_RAM_FUNC static void ssm_block_step(
		const ssm_block_weights_t *w,
		float h[SSM_D_INNER][SSM_D_STATE],
		float conv_hist[SSM_D_INNER][SSM_D_CONV - 1],
		const float *x_norm,
		float *block_out) {

	/* Reciprocal hoist: at most 4 divisions per block call, instead of
	 * one per element at every quantized boundary. */
	ssm_act_scales_t q;
	ssm_get_act_scales(w, &q);
	ssm_qscale_finish(&q.z);
	ssm_qscale_finish(&q.u);
	ssm_qscale_finish(&q.y_gated);
	ssm_qscale_finish(&q.block_out);

	float u_raw[SSM_D_INNER];
	float z[SSM_D_INNER];
	float u[SSM_D_INNER];
	float y[SSM_D_INNER];

	{
		int half = SSM_D_INNER / 2;
		ssm_channel_ctx_t ctx1 = { w, x_norm, conv_hist, h, &q, half, SSM_D_INNER, u_raw, z, u, y };
		ssm_dispatch_to_core1(ssm_channel_work, &ctx1);

		ssm_channel_ctx_t ctx0 = { w, x_norm, conv_hist, h, &q, 0, half, u_raw, z, u, y };
		ssm_channel_work(&ctx0);

		ssm_wait_for_core1();
	}

	{
		int half = SSM_D_MODEL / 2;
		ssm_proj_ctx_t ctx1 = { w, y, half, SSM_D_MODEL, block_out };
		ssm_dispatch_to_core1(ssm_out_proj_work, &ctx1);

		ssm_out_proj_range(w, y, 0, half, block_out);

		ssm_wait_for_core1();
	}
	for (int o = 0; o < SSM_D_MODEL; o++) {
		block_out[o] = ssm_qdq_r(block_out[o], &q.block_out);
	}
}

SSM_RAM_FUNC void SSMBackbone_Reset(SSMBackbone_State *state) {
	memset(state, 0, sizeof(*state));
}

SSM_RAM_FUNC void SSMBackbone_ProcessFrame(SSMBackbone_State *state, const float *frame_in, float *final_norm_out) {
	float x[SSM_D_MODEL];
	float x_norm[SSM_D_MODEL];
	float block_out[SSM_D_MODEL];

	memcpy(x, frame_in, sizeof(x));

	for (int layer = 0; layer < SSM_N_LAYERS; layer++) {
		ssm_rmsnorm(x, &ssm_blocks[layer].norm_w, x_norm);
		ssm_block_step(&ssm_blocks[layer], state->h[layer], state->conv_hist[layer], x_norm, block_out);
		for (int i = 0; i < SSM_D_MODEL; i++) {
			x[i] += block_out[i];
		}
	}

	float normed[SSM_D_MODEL];
	ssm_rmsnorm(x, &ssm_final_norm_w, normed);

	ssm_qscale_t q_final;
	q_final.s = ssm_get_final_norm_scale();
	ssm_qscale_finish(&q_final);
	for (int i = 0; i < SSM_D_MODEL; i++) {
		normed[i] = ssm_qdq_r(normed[i], &q_final);
	}

	for (int i = 0; i < SSM_D_MODEL; i++) {
		state->pooled_sum[i] += normed[i];
	}
	state->frame_count++;

	if (final_norm_out != NULL) {
		memcpy(final_norm_out, normed, sizeof(normed));
	}
}

SSM_RAM_FUNC void SSMBackbone_GetPooled(const SSMBackbone_State *state, float *pooled_out) {
	float inv_n = 1.0f / (float) state->frame_count;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		pooled_out[i] = state->pooled_sum[i] * inv_n;
	}
}

SSM_RAM_FUNC void SSMBackbone_NormalizeFrame(const float *raw, float *normalized) {
	for (int i = 0; i < SSM_D_MODEL; i++) {
		normalized[i] = (raw[i] - ssm_norm_mean[i]) / ssm_norm_std[i];
	}
}