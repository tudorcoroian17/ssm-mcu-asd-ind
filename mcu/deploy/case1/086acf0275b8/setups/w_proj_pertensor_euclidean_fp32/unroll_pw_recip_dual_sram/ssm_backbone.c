#include "ssm_backbone.h"

#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------------
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

/* ---------------------------------------------------------------------
 * Activation quantize-dequantize, reciprocal-hoisted.
 *
 * Claude comment: the generated header defines each SSM_QUANT_* hook in
 * one of exactly two forms:
 *   - no activation quantization:  #define SSM_QUANT_U(w, val) (val)
 *   - actboundaries:               #define SSM_QUANT_U(w, val)
 *                                      (ssm_quant_dequant((val), (w)->u_scale))
 * The u_scale / z_scale / y_gated_scale / block_out_scale fields (and the
 * global ssm_final_norm_scale) EXIST ONLY in the actboundaries headers, so
 * this file cannot name them directly -- it would not compile for the
 * other setups.
 *
 * The trick below gets the scale out THROUGH the hook macro instead. For
 * the extraction functions only, ssm_quant_dequant is a macro that returns
 * its second argument. The hooks are called with a -1.0f sentinel:
 *   - actboundaries form expands to  ((w)->u_scale)  -> the real scale (> 0)
 *   - identity form expands to       (-1.0f)         -> "not quantized"
 * The macro is removed right after, so nothing else in the file sees it.
 * This file never defines a ssm_quant_dequant function.
 *
 * NOT bit-exact under actboundaries: roundf(v * inv_s) can differ from
 * roundf(v / s) by one LSB when v / s lands next to a .5 boundary. Same
 * trade-off as the true_int8 reciprocal hoist. Bit-exact for every other
 * setup (the hook is skipped, exactly like the (val) no-op).
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
 * Matvecs: one per weight matrix, because each matrix has its own
 * SSM_*_W macro.
 *
 * Claude comment: weights go through the macro, NOT a raw pointer. The
 * macro hides whether the scale is scale[0] (pertensor) or scale[r]
 * (perchannel), and the scale array is 1 entry long for pertensor, so a
 * raw walk would read out of bounds. Only the input vector is
 * pointer-walked. The weight row r is fixed inside the inner loop, so the
 * compiler hoists the row base and the scale load out of it.
 *
 * Unrolled 4x. Bit-exact: every term is the same expression as the
 * original loop, (dequantized weight) * x, added to ONE accumulator in
 * the same left-to-right order. The tail loop is dead for the current
 * dims (all multiples of 4); it stays for safety.
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

SSM_RAM_FUNC static void ssm_x_proj_range(const ssm_block_weights_t *w,
		const float *x, int o_start, int o_end, float *out) {
	for (int o = o_start; o < o_end; o++) {
		const float *x_p = x;
		float acc = 0.0f;
		int i = 0;
		while (i + 4 <= SSM_D_INNER) {
			acc += SSM_X_PROJ_W(w, o, i) * x_p[0];
			acc += SSM_X_PROJ_W(w, o, i + 1) * x_p[1];
			acc += SSM_X_PROJ_W(w, o, i + 2) * x_p[2];
			acc += SSM_X_PROJ_W(w, o, i + 3) * x_p[3];
			x_p += 4;
			i += 4;
		}
		while (i < SSM_D_INNER) {
			acc += SSM_X_PROJ_W(w, o, i) * (*x_p++);
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

/* --- x_proj / out_proj: whole-vector ops, split by output row. --- */

typedef struct {
	const ssm_block_weights_t *w;
	const float *x;
	int o_start;
	int o_end;
	float *out;
} ssm_proj_ctx_t;

SSM_RAM_FUNC static void ssm_x_proj_work(void *ctx_v) {
	ssm_proj_ctx_t *ctx = (ssm_proj_ctx_t *) ctx_v;
	ssm_x_proj_range(ctx->w, ctx->x, ctx->o_start, ctx->o_end, ctx->out);
}

SSM_RAM_FUNC static void ssm_out_proj_work(void *ctx_v) {
	ssm_proj_ctx_t *ctx = (ssm_proj_ctx_t *) ctx_v;
	ssm_out_proj_range(ctx->w, ctx->x, ctx->o_start, ctx->o_end, ctx->out);
}

/* --- Phase 1: in_proj_u, in_proj_z, and the conv step, split by
 * channel. Each channel's conv only touches its own conv_hist[c] and
 * u_raw[c] -- no cross-channel dependency. The z and u hooks are
 * element-wise, so applying them per channel inside each core's range
 * gives the same values as the original "whole array, then quantize"
 * order. --- */

typedef struct {
	const ssm_block_weights_t *w;
	const float *x_norm;
	float (*conv_hist)[SSM_D_CONV - 1];
	const ssm_act_scales_t *q;
	int c_start;
	int c_end;
	float *u_raw;
	float *z;
	float *u;
} ssm_phase1_ctx_t;

SSM_RAM_FUNC static void ssm_phase1_work(void *ctx_v) {
	ssm_phase1_ctx_t *ctx = (ssm_phase1_ctx_t *) ctx_v;
	const ssm_block_weights_t *w = ctx->w;

	ssm_in_proj_range(w, 0, ctx->x_norm, ctx->c_start, ctx->c_end, ctx->u_raw);
	ssm_in_proj_range(w, SSM_D_INNER, ctx->x_norm, ctx->c_start, ctx->c_end, ctx->z);

	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		ctx->z[c] = ssm_qdq_r(ctx->z[c], &ctx->q->z);
	}

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
}

/* --- Phase 3: dt_proj (folded in -- SSM_DT_RANK is too small to
 * dispatch on its own, same reasoning as true_int8) + euler-discretize +
 * recurrence + y_gated, split by channel. h[c] carries no cross-channel
 * dependency.
 *
 * Claude comment: dt_proj's row-matvec and the state scan stay as plain
 * macro loops, not unrolled -- same known gap as the true_int8 and
 * full_fp32 backbones, not a regression here. --- */

typedef struct {
	const ssm_block_weights_t *w;
	float (*h)[SSM_D_STATE];
	const float *delta_low;
	const float *B;
	const float *C;
	const float *u;
	const float *z;
	const ssm_act_scales_t *q;
	int c_start;
	int c_end;
	float *y;
} ssm_phase3_ctx_t;

SSM_RAM_FUNC static void ssm_phase3_work(void *ctx_v) {
	ssm_phase3_ctx_t *ctx = (ssm_phase3_ctx_t *) ctx_v;
	const ssm_block_weights_t *w = ctx->w;

	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		float acc = SSM_DT_PROJ_B(w, c);
		for (int i = 0; i < SSM_DT_RANK; i++) {
			acc += SSM_DT_PROJ_W(w, c, i) * ctx->delta_low[i];
		}
		float delta_c = ssm_softplus(acc);

		float y_c = 0.0f;
		for (int n = 0; n < SSM_D_STATE; n++) {
			float deltaA = delta_c * SSM_A(w, c, n);
			if (deltaA < -1.9f) {
				deltaA = -1.9f;
			}
			float A_bar = 1.0f + deltaA;
			float B_bar = delta_c * ctx->B[n];
			float new_h = A_bar * ctx->h[c][n] + B_bar * ctx->u[c];
			ctx->h[c][n] = new_h;
			y_c += new_h * ctx->C[n];
		}
		float y_val = (y_c + ctx->u[c] * SSM_D_PARAM(w, c)) * ssm_silu(ctx->z[c]);
		ctx->y[c] = ssm_qdq_r(y_val, &ctx->q->y_gated);
	}
}

/* Claude comment: 4 dual-core round trips per block -- Phase 1, x_proj,
 * Phase 3, out_proj -- same shape as the true_int8 and full_fp32
 * backbones, for the same reason (each is a real data-dependency
 * barrier). */
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

	{
		int half = SSM_D_INNER / 2;
		ssm_phase1_ctx_t ctx1 = { w, x_norm, conv_hist, &q, half, SSM_D_INNER, u_raw, z, u };
		ssm_dispatch_to_core1(ssm_phase1_work, &ctx1);

		ssm_phase1_ctx_t ctx0 = { w, x_norm, conv_hist, &q, 0, half, u_raw, z, u };
		ssm_phase1_work(&ctx0);

		ssm_wait_for_core1();
	}

	float x_dbl[SSM_DT_RANK + 2 * SSM_D_STATE];
	{
		const int out_dim = SSM_DT_RANK + 2 * SSM_D_STATE;
		int half = out_dim / 2;
		ssm_proj_ctx_t ctx1 = { w, u, half, out_dim, x_dbl };
		ssm_dispatch_to_core1(ssm_x_proj_work, &ctx1);

		ssm_x_proj_range(w, u, 0, half, x_dbl);

		ssm_wait_for_core1();
	}

	float delta_low[SSM_DT_RANK];
	float B[SSM_D_STATE];
	float C[SSM_D_STATE];
	memcpy(delta_low, &x_dbl[0], sizeof(delta_low));
	memcpy(B, &x_dbl[SSM_DT_RANK], sizeof(B));
	memcpy(C, &x_dbl[SSM_DT_RANK + SSM_D_STATE], sizeof(C));

	float y[SSM_D_INNER];
	{
		int half = SSM_D_INNER / 2;
		ssm_phase3_ctx_t ctx1 = { w, h, delta_low, B, C, u, z, &q, half, SSM_D_INNER, y };
		ssm_dispatch_to_core1(ssm_phase3_work, &ctx1);

		ssm_phase3_ctx_t ctx0 = { w, h, delta_low, B, C, u, z, &q, 0, half, y };
		ssm_phase3_work(&ctx0);

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