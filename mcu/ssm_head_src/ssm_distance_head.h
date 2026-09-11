#pragma once

#include <stdint.h>
#include "ssm_head_ref.h"

/* Scheme-agnostic: takes whatever pooled embedding a backbone produced
 * (float32, SSM_D_MODEL entries -- the fake-quant backbones' embedding is
 * already dequantized to float on the way out; the true-int8 backbone's
 * embedding leaves as an int8-round-tripped float, same interface either
 * way) and scores it against the reference vectors this scheme's export
 * baked into ssm_head_ref.c. Never generated -- copied byte-for-byte into
 * every deploy folder by export_reference_heads.py, same convention as
 * ssm_backbone_src's canonical backbone. */

typedef struct {
	float euclidean_score;
	float knn16_score;
	uint8_t euclidean_anomaly; /* 1 if euclidean_score > ssm_threshold_euclidean */
	uint8_t knn16_anomaly;     /* 1 if knn16_score > ssm_threshold_knn16 */
} SSMHeadResult;

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result);