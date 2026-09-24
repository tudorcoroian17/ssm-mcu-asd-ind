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

SSM_RAM_FUNC static void ssm_rmsnorm(const float *x, const ssm_norm_weights_t *nw, float *out) {
	float ss = 0.0f;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		ss += x[i] * x[i];
	}
	/* Claude comment: recip-hoist -- the only division this scheme has.
	 * Every SSM_QUANT_* macro in this scheme's generated header is a
	 * no-op ((val), nothing to requantize), so there's nothing else to
	 * hoist. */
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

/* Claude comment: row-major float matvec, out[o] written directly (not
 * offset), matching ssm_qlinear_real_range's convention in the true_int8
 * files. Reaches past the SSM_*_W macros into the raw weight array --
 * safe ONLY in this file, because this scheme's weights are confirmed
 * plain float with no scale (checked against the generated header, not
 * assumed). Unrolled 4x, pointer-walked on both operands. Bit-exact:
 * same left-to-right summation order as the original macro-per-element
 * loop, just restructured indexing. */
SSM_RAM_FUNC static void ssm_linear_range(const float *w_mat, const float *x,
		int row_start, int row_end, int in_dim, float *out) {
	for (int o = row_start; o < row_end; o++) {
		const float *w_row = &w_mat[o * in_dim];
		const float *x_p = x;
		float acc = 0.0f;
		int i = 0;
		while (i + 4 <= in_dim) {
			acc += w_row[0] * x_p[0];
			acc += w_row[1] * x_p[1];
			acc += w_row[2] * x_p[2];
			acc += w_row[3] * x_p[3];
			w_row += 4;
			x_p += 4;
			i += 4;
		}
		while (i < in_dim) {
			acc += (*w_row++) * (*x_p++);
			i++;
		}
		out[o] = acc;
	}
}

/* --- x_proj / out_proj: whole-vector ops, split by output row. --- */

typedef struct {
	const float *w_mat;
	const float *x;
	int row_start;
	int row_end;
	int in_dim;
	float *out;
} ssm_linear_ctx_t;

SSM_RAM_FUNC static void ssm_linear_work(void *ctx_v) {
	ssm_linear_ctx_t *ctx = (ssm_linear_ctx_t *) ctx_v;
	ssm_linear_range(ctx->w_mat, ctx->x, ctx->row_start, ctx->row_end, ctx->in_dim, ctx->out);
}

SSM_RAM_FUNC static void ssm_linear_parallel(const float *w_mat, const float *x,
		int out_dim, int in_dim, float *out) {
	int half = out_dim / 2;
	ssm_linear_ctx_t ctx1 = { w_mat, x, half, out_dim, in_dim, out };
	ssm_dispatch_to_core1(ssm_linear_work, &ctx1);

	ssm_linear_range(w_mat, x, 0, half, in_dim, out);

	ssm_wait_for_core1();
}

/* --- Phase 1: in_proj_u, in_proj_z, and the conv step, split by
 * channel. Each channel's conv only touches its own conv_hist[c] and
 * u_raw[c] -- no cross-channel dependency.
 *
 * Claude comment: in_proj is ONE weight matrix here (2*D_INNER rows: u
 * is rows [0,D_INNER), z is rows [D_INNER,2*D_INNER)) -- unlike
 * true_int8, which has two separate matrices. z's call below passes a
 * shifted base pointer (&w->in_proj_w[D_INNER*D_MODEL]) instead of an
 * output offset, so ssm_linear_range's out[o] convention still writes
 * z[c] directly. --- */

typedef struct {
	const ssm_block_weights_t *w;
	const float *x_norm;
	float (*conv_hist)[SSM_D_CONV - 1];
	int c_start;
	int c_end;
	float *u_raw;
	float *z;
	float *u;
} ssm_phase1_ctx_t;

SSM_RAM_FUNC static void ssm_phase1_work(void *ctx_v) {
	ssm_phase1_ctx_t *ctx = (ssm_phase1_ctx_t *) ctx_v;
	const ssm_block_weights_t *w = ctx->w;

	ssm_linear_range(w->in_proj_w, ctx->x_norm, ctx->c_start, ctx->c_end, SSM_D_MODEL, ctx->u_raw);
	ssm_linear_range(&w->in_proj_w[SSM_D_INNER * SSM_D_MODEL], ctx->x_norm,
			ctx->c_start, ctx->c_end, SSM_D_MODEL, ctx->z);

	for (int c = ctx->c_start; c < ctx->c_end; c++) {
		ctx->z[c] = SSM_QUANT_Z(w, ctx->z[c]);
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
		ctx->u[c] = SSM_QUANT_U(w, ctx->u[c]);
	}
}

/* --- Phase 3: dt_proj (folded in -- SSM_DT_RANK is too small to
 * dispatch on its own, same reasoning as true_int8) + euler-discretize +
 * recurrence + y_gated, split by channel. h[c] carries no cross-channel
 * dependency.
 *
 * Claude comment: dt_proj's own row-matvec is left as plain macro calls,
 * not unrolled -- SSM_DT_RANK is small and this matches the true_int8
 * backbone's current state (its dt_proj loop isn't unrolled either;
 * that's a known, not-yet-done gap there too, not a regression here). */

typedef struct {
	const ssm_block_weights_t *w;
	float (*h)[SSM_D_STATE];
	const float *delta_low;
	const float *B;
	const float *C;
	const float *u;
	const float *z;
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
		ctx->y[c] = (y_c + ctx->u[c] * SSM_D_PARAM(w, c)) * ssm_silu(ctx->z[c]);
		ctx->y[c] = SSM_QUANT_Y_GATED(w, ctx->y[c]);
	}
}

/* Claude comment: 4 dual-core round trips per block -- Phase 1, x_proj,
 * Phase 3, out_proj -- same shape as the true_int8 backbone, for the
 * same reason (each is a genuine data-dependency barrier). */
SSM_RAM_FUNC static void ssm_block_step(
		const ssm_block_weights_t *w,
		float h[SSM_D_INNER][SSM_D_STATE],
		float conv_hist[SSM_D_INNER][SSM_D_CONV - 1],
		const float *x_norm,
		float *block_out) {

	float u_raw[SSM_D_INNER];
	float z[SSM_D_INNER];
	float u[SSM_D_INNER];

	{
		int half = SSM_D_INNER / 2;
		ssm_phase1_ctx_t ctx1 = { w, x_norm, conv_hist, half, SSM_D_INNER, u_raw, z, u };
		ssm_dispatch_to_core1(ssm_phase1_work, &ctx1);

		ssm_phase1_ctx_t ctx0 = { w, x_norm, conv_hist, 0, half, u_raw, z, u };
		ssm_phase1_work(&ctx0);

		ssm_wait_for_core1();
	}

	float x_dbl[SSM_DT_RANK + 2 * SSM_D_STATE];
	ssm_linear_parallel(w->x_proj_w, u, SSM_DT_RANK + 2 * SSM_D_STATE, SSM_D_INNER, x_dbl);

	float delta_low[SSM_DT_RANK];
	float B[SSM_D_STATE];
	float C[SSM_D_STATE];
	memcpy(delta_low, &x_dbl[0], sizeof(delta_low));
	memcpy(B, &x_dbl[SSM_DT_RANK], sizeof(B));
	memcpy(C, &x_dbl[SSM_DT_RANK + SSM_D_STATE], sizeof(C));

	float y[SSM_D_INNER];
	{
		int half = SSM_D_INNER / 2;
		ssm_phase3_ctx_t ctx1 = { w, h, delta_low, B, C, u, z, half, SSM_D_INNER, y };
		ssm_dispatch_to_core1(ssm_phase3_work, &ctx1);

		ssm_phase3_ctx_t ctx0 = { w, h, delta_low, B, C, u, z, 0, half, y };
		ssm_phase3_work(&ctx0);

		ssm_wait_for_core1();
	}

	ssm_linear_parallel(w->out_proj_w, y, SSM_D_MODEL, SSM_D_INNER, block_out);
	for (int o = 0; o < SSM_D_MODEL; o++) {
		block_out[o] = SSM_QUANT_BLOCK_OUT(w, block_out[o]);
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
	for (int i = 0; i < SSM_D_MODEL; i++) {
		normed[i] = SSM_QUANT_FINAL_NORM(normed[i]);
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