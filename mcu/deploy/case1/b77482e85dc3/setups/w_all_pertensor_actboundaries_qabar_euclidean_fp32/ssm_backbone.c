#include "ssm_backbone.h"

#include <math.h>
#include <string.h>

/* CLASSIC (selective=False) backbone. Differences from the selective
 * ssm_backbone.c, all of them consequences of ssm_block.py's `else` arm:
 *
 *   - No x_proj, no dt_proj, no split. delta, B and C are static
 *     parameters, so nothing is projected out of u per frame.
 *   - The recurrence coefficients reach this file through SSM_DELTA,
 *     SSM_A_BAR and SSM_B_BAR, defined in the GENERATED ssm_weights.h.
 *     Under the qab schemes those macros expand to the live discretize
 *     math below; under the qabar schemes they read baked constant arrays
 *     and discard delta_c. This file's text is identical either way --
 *     same convention as SSM_A, which already hides whether A is stored
 *     precomputed or as a quantized A_log needing -expf on each access.
 *
 * Everything else -- in_proj, the depthwise conv, the D shortcut, the z
 * gate, out_proj, RMSNorm, pooling, and every SSM_QUANT_* hook -- is
 * unchanged from the selective source, deliberately, so the two can be
 * diffed. */

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

static inline float ssm_quant_dequant(float v, float scale) {
	float q = roundf(v / scale);
	if (q > 127.0f) q = 127.0f;
	if (q < -127.0f) q = -127.0f;
	return q * scale;
}

static void ssm_rmsnorm(const float *x, const ssm_norm_weights_t *nw, float *out) {
	float ss = 0.0f;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		ss += x[i] * x[i];
	}
	float inv_rms = 1.0f / sqrtf(ss / (float) SSM_D_MODEL + 1e-5f);
	for (int i = 0; i < SSM_D_MODEL; i++) {
		out[i] = x[i] * inv_rms * SSM_NORM_W(nw, i);
	}
}

static void ssm_block_step(
		const ssm_block_weights_t *w,
		float h[SSM_D_INNER][SSM_D_STATE],
		float conv_hist[SSM_D_INNER][SSM_D_CONV - 1],
		const float *x_norm,
		float *block_out) {

	float u_raw[SSM_D_INNER];
	float z[SSM_D_INNER];
	float u[SSM_D_INNER];
	float y[SSM_D_INNER];

	/* in_proj: split into u_raw (rows 0..D_INNER-1) and z (rows D_INNER..).
	 * SSM_QUANT_Z is applied once z[] is fully computed -- boundaries group,
	 * no-op when this scheme doesn't quantize activations. */
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = 0.0f;
		for (int i = 0; i < SSM_D_MODEL; i++) {
			acc += SSM_IN_PROJ_W(w, o, i) * x_norm[i];
		}
		u_raw[o] = acc;
	}
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = 0.0f;
		for (int i = 0; i < SSM_D_MODEL; i++) {
			acc += SSM_IN_PROJ_W(w, SSM_D_INNER + o, i) * x_norm[i];
		}
		z[o] = acc;
	}
	for (int o = 0; o < SSM_D_INNER; o++) {
		z[o] = SSM_QUANT_Z(w, z[o]);
	}

	/*	causal depthwise conv: kernel tap 0 = oldest (t-3) .. tap 3 = current
	 * 	(t), then SiLU. conv_hist holds t-3,t-2,t-1 oldest-first.
	 * 	SSM_QUANT_U applied once u[] (post-conv, post-SiLU -- Python's
	 * 	u_post_conv_silu) is fully computed. */
	for (int c = 0; c < SSM_D_INNER; c++) {
		float acc = SSM_CONV_B(w, c);
		for (int k = 0; k < SSM_D_CONV - 1; k++) {
			acc += SSM_CONV_W(w, c, k) * conv_hist[c][k];
		}
		acc += SSM_CONV_W(w, c, SSM_D_CONV - 1) * u_raw[c];
		u[c] = ssm_silu(acc);

		for (int k = 0; k < SSM_D_CONV - 2; k++) {
			conv_hist[c][k] = conv_hist[c][k + 1];
		}
		conv_hist[c][SSM_D_CONV - 2] = u_raw[c];
	}
	for (int c = 0; c < SSM_D_INNER; c++) {
		u[c] = SSM_QUANT_U(w, u[c]);
	}

	/* 	Recurrence + output, fused per channel.
	 *
	 * 	delta_c is hoisted out of the inner loop deliberately: under qab it
	 * 	is softplus(dequantized dt[c]), which must not be recomputed
	 * 	SSM_D_STATE times per channel. Under qabar it is 0.0f and both
	 * 	coefficient macros discard it via the comma operator, which is also
	 * 	what keeps -Wunused-variable quiet there.
	 *
	 * 	y[c] here is Python's y_gated (post D-shortcut, post SiLU gate) --
	 * 	SSM_QUANT_Y_GATED applies right after it's computed. y_c (the raw
	 * 	scan output, pre-shortcut-and-gate -- Python's y_scan) has no
	 * 	quantize hook: it's a scalar consumed on the same line it's
	 * 	produced, not a materialized array. That's scan-group work, not
	 * 	boundaries -- deliberately not wired yet, same as the selective
	 * 	source. */
	for (int c = 0; c < SSM_D_INNER; c++) {
		const float delta_c = SSM_DELTA(w, c);
		float y_c = 0.0f;
		for (int n = 0; n < SSM_D_STATE; n++) {
			float A_bar = SSM_A_BAR(w, c, n, delta_c);
			float B_bar = SSM_B_BAR(w, c, n, delta_c);
			float new_h = A_bar * h[c][n] + B_bar * u[c];
			h[c][n] = new_h;
			y_c += new_h * SSM_C(w, n);
		}
		y[c] = (y_c + u[c] * SSM_D_PARAM(w, c)) * ssm_silu(z[c]);
		y[c] = SSM_QUANT_Y_GATED(w, y[c]);
	}

	for (int o = 0; o < SSM_D_MODEL; o++) {
		float acc = 0.0f;
		for (int i = 0; i < SSM_D_INNER; i++) {
			acc += SSM_OUT_PROJ_W(w, o, i) * y[i];
		}
		block_out[o] = acc;
	}
	for (int o = 0; o < SSM_D_MODEL; o++) {
		block_out[o] = SSM_QUANT_BLOCK_OUT(w, block_out[o]);
	}
}

void SSMBackbone_Reset(SSMBackbone_State *state) {
	memset(state, 0, sizeof(*state));
}

void SSMBackbone_ProcessFrame(SSMBackbone_State *state, const float *frame_in, float *final_norm_out) {
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
		state -> pooled_sum[i] += normed[i];
	}
	state->frame_count++;

	if (final_norm_out != NULL) {
		memcpy(final_norm_out, normed, sizeof(normed));
	}
}

void SSMBackbone_GetPooled(const SSMBackbone_State *state, float *pooled_out) {
	float inv_n = 1.0f / (float) state->frame_count;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		pooled_out[i] = state->pooled_sum[i] * inv_n;
	}
}

void SSMBackbone_NormalizeFrame(const float *raw, float *normalized) {
    for (int i = 0; i < SSM_D_MODEL; i++) {
        normalized[i] = (raw[i] - ssm_norm_mean[i]) / ssm_norm_std[i];
    }
}