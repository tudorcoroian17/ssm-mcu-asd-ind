#pragma once

#include <stdint.h>
#include "ssm_weights.h"

/* FULL_FP32 backbone: every weight is a plain float array, no
 * quantization anywhere. */

void ssm_core1_loop_step(void);

typedef struct {
	float h[SSM_N_LAYERS][SSM_D_INNER][SSM_D_STATE];
	float conv_hist[SSM_N_LAYERS][SSM_D_INNER][SSM_D_CONV - 1]; /* oldest first */
	float pooled_sum[SSM_D_MODEL];
	uint32_t frame_count;
} SSMBackbone_State;

void SSMBackbone_Reset(SSMBackbone_State *state);

/*	Runs one log-mel frame (SSM_D_MODEL floats -- n_mels here, since
 *	learned_input_embed is false) through the full stacked backbone and
 *	folds it into the pooling sum. Call once per frame, in clip order.
 * 	final_norm_out may be NULL if you only need the pooled embedding. */
void SSMBackbone_ProcessFrame(SSMBackbone_State *state, const float *frame_in, float *final_norm_out);

/* Valid only after at least one frame. */
void SSMBackbone_GetPooled(const SSMBackbone_State *state, float *pooled_out);

/* Applies this model's per-channel z-score normalization (trained-fold
 * mean/std) to one raw log-mel frame. Call BEFORE SSMBackbone_ProcessFrame
 * -- the model was trained on normalized input, not raw log-mel. */
void SSMBackbone_NormalizeFrame(const float *raw, float *normalized);