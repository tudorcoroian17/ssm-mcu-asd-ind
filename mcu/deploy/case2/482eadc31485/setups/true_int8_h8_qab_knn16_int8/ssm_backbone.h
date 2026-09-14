#pragma once

#include <stdint.h>
#include "ssm_weights.h"

/* CLASSIC (selective=False) backbone, true int8 arithmetic. */

/* h storage width -- the one scheme-specific arithmetic choice, flipped by
 * a single #define set (or not) in the GENERATED ssm_weights.h. Everything
 * else in this file and ssm_backbone.c is identical for both widths, and
 * identical for both recurrence forms: scheme-specific decisions live in
 * the generated header, never in the hand-maintained source. */
#ifdef SSM_H_WIDTH_INT16
typedef int16_t ssm_h_t;
#define SSM_H_CLIP 32767
#else
typedef int8_t ssm_h_t;
#define SSM_H_CLIP 127
#endif

/* ssm_true_int8_layer_t, the SSM_TI_* recurrence macros, and the extern
 * data declarations live in the GENERATED ssm_weights.h, not here -- same
 * one-directional include convention every other scheme uses. ssm_weights.h
 * is included above specifically to bring them into scope. */

typedef struct {
	ssm_h_t h[SSM_N_LAYERS][SSM_D_INNER][SSM_D_STATE];
	/* Always int8 range regardless of h-width: this stores u_raw_q, which
	 * is quantized to s_conv_out's int8 domain, never to h's domain. */
	int8_t conv_hist_q[SSM_N_LAYERS][SSM_D_INNER][SSM_D_CONV - 1]; /* oldest first */
	float pooled_sum[SSM_D_MODEL]; /* real units -- accumulates
	                                 * normed_q * ssm_final_norm_scale each
	                                 * frame, matching quantized_forward_classic's
	                                 * pooled_sum exactly. */
	uint32_t frame_count;
} SSMBackbone_State;

void SSMBackbone_Reset(SSMBackbone_State *state);

/* Runs one ALREADY-NORMALIZED log-mel frame (SSM_D_MODEL floats) through
 * the full stacked backbone using genuine int32-accumulate arithmetic, and
 * folds the result into the pooling sum. Call once per frame, in clip
 * order. final_norm_out may be NULL if only the pooled embedding is needed;
 * when non-NULL, it receives this frame's int8-round-tripped normed output
 * (float, but exactly representable as int8 * ssm_final_norm_scale). */
void SSMBackbone_ProcessFrame(SSMBackbone_State *state, const float *frame_in, float *final_norm_out);

/* Valid only after at least one frame. Matches quantized_forward_classic's
 * final step exactly: the mean is itself re-quantized to
 * ssm_final_norm_scale before being returned. */
void SSMBackbone_GetPooled(const SSMBackbone_State *state, float *pooled_out);

/* Same per-channel z-score normalization as every other scheme -- apply
 * BEFORE SSMBackbone_ProcessFrame. */
void SSMBackbone_NormalizeFrame(const float *raw, float *normalized);