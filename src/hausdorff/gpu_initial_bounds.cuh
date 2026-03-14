#pragma once
#include <cuda_runtime.h>
#include "bvh/lbvh.cuh"

// ─── Result of GPU initial bound estimation ──────────────────────────────────
struct InitialBounds {
    double L;   // global lower bound (squared)
    double U;   // global upper bound (squared)
};

// ─── Compute initial L and U in parallel ─────────────────────────────────────
// For each triangle in mesh A:
//   1. Find closest triangle in mesh B via LBVH traversal
//   2. Compute point-to-triangle distances for all 3 vertices + centroid
//   3. Derive local L and U; reduce globally via atomicMax/atomicMin
//
// tris_A[nA], tris_B[nB] — float3x3 (same layout as lbvh.cuh)
// lbvh_B — pre-built LBVH over mesh B
// out — device pointer to InitialBounds (must be pre-allocated and zeroed)
void gpu_compute_initial_bounds(
    const float3x3* d_tris_A, int nA,
    const float3x3* d_tris_B, int nB,
    const LBVH& lbvh_B,
    InitialBounds* d_out);
