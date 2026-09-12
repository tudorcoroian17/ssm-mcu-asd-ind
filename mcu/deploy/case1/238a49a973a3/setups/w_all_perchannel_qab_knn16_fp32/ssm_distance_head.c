#include "ssm_distance_head.h"

#include <math.h>
#include "ssm_head_ref.h"

static float ssm_l2_distance(const float *a, const float *b, int n) {
	float acc = 0.0f;
	for (int i = 0; i < n; i++) {
		float d = a[i] - b[i];
		acc += d * d;
	}
	return sqrtf(acc);
}

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result) {
	float best = INFINITY;
	for (int k = 0; k < SSM_KNN16_N_CLUSTERS; k++) {
		float d = ssm_l2_distance(embedding, ssm_ref_clusters[k], SSM_D_MODEL);
		if (d < best) {
			best = d;
		}
	}
	result->knn16_score = best;
	result->knn16_anomaly = (result->knn16_score > ssm_threshold_knn16) ? 1 : 0;

	/* euclidean not scored in this folder -- sentinel. */
	result->euclidean_score = -1.0f;
	result->euclidean_anomaly = 0;
}
