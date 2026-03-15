#pragma once
#include <cuda_runtime.h>
#include "bvh/lbvh.cuh"

// Result for one triangle: squared distances from 3 vertices to nearest triangle on B.
struct TriQueryResult {
    double d[3];  // d[k] = squared dist from vertex k to nearest triangle on B
};

// Batch query: for each triangle in h_tris (nA triangles, host pointer),
// find nearest triangle on B via LBVH and return per-vertex squared distances.
// Results written to h_out (host pointer, pre-allocated, size nA).
void gpu_batch_query(
    const float3x3* h_tris, int nA,
    const LBVH& lbvh_B,
    const float3x3* d_tris_B,
    TriQueryResult* h_out);
