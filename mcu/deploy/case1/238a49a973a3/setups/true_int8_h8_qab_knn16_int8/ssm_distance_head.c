#include "ssm_distance_head.h"

#include <math.h>
#include "ssm_weights.h"
#include "ssm_head_ref.h"

static int8_t ssm_head_quantize(float real) {
	float q = roundf(real / ssm_final_norm_scale);
	if (q > 127.0f) {
		q = 127.0f;
	}
	if (q < -127.0f) {
		q = -127.0f;
	}
	return (int8_t) q;
}

static int32_t ssm_sumsq(const int8_t *a, const int8_t *b, int n) {
	int32_t acc = 0;
	for (int i = 0; i < n; i++) {
		int32_t d = (int32_t) a[i] - (int32_t) b[i];
		acc += d * d;
	}
	return acc;
}

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result) {
	int8_t embedding_q[SSM_D_MODEL];
	for (int i = 0; i < SSM_D_MODEL; i++) {
		embedding_q[i] = ssm_head_quantize(embedding[i]);
	}

	int32_t best = -1;
	for (int k = 0; k < SSM_KNN16_N_CLUSTERS; k++) {
		int32_t sumsq = ssm_sumsq(embedding_q, ssm_ref_clusters_q[k], SSM_D_MODEL);
		if (best < 0 || sumsq < best) {
			best = sumsq;
		}
	}
	result->knn16_sumsq = best;
	result->knn16_score = sqrtf((float) result->knn16_sumsq) * ssm_final_norm_scale;
	result->knn16_anomaly = (result->knn16_sumsq > ssm_threshold_knn16_sumsq) ? 1 : 0;

	/* euclidean not scored in this folder -- sentinel. */
	result->euclidean_sumsq = -1;
	result->euclidean_score = -1.0f;
	result->euclidean_anomaly = 0;
}
