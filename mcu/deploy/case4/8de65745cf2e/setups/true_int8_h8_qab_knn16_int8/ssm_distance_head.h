#pragma once

#include <stdint.h>
#include "ssm_weights.h"

/* SINGLE-HEAD, pure-int32 build: this folder scores knn_clustered_16 only, with
 * the entire decision path in int32 (only the reported score uses a
 * cosmetic sqrt). The other head's fields are a sentinel. SSMHeadResult
 * keeps all four fields so main.c is identical across every setup. Valid
 * ONLY on a true-int8 backbone whose pooled embedding is an exact int8
 * round trip at ssm_final_norm_scale. */

#define SSM_KNN16_N_CLUSTERS 16

typedef struct {
	float euclidean_score;
	float knn16_score;
	uint8_t euclidean_anomaly;
	uint8_t knn16_anomaly;
	int32_t euclidean_sumsq;
	int32_t knn16_sumsq;
} SSMHeadResult;

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result);
