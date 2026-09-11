#pragma once

#include <stdint.h>
#include "ssm_weights.h"

#define SSM_KNN16_N_CLUSTERS 16

extern const int8_t ssm_ref_centroid_q[SSM_D_MODEL];
extern const int8_t ssm_ref_clusters_q[SSM_KNN16_N_CLUSTERS][SSM_D_MODEL];
extern const int32_t ssm_threshold_euclidean_sumsq;
extern const int32_t ssm_threshold_knn16_sumsq;
