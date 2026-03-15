#pragma once
#include <cuda_runtime.h>
#include "bvh/lbvh.cuh"

// Result for one triangle: squared distances and nearest prim index for each vertex.
struct TriQueryResult {
    double d[3];        // squared dist from vertex k to nearest triangle on B
    int    nearest[3];  // index of nearest triangle on B for each vertex
};

// Batch query: for nA triangles in h_tris (host), find nearest triangle on B via LBVH.
// d_tris_B must be pre-uploaded to device (persistent across calls).
// h_out: host output, pre-allocated, size nA.
void gpu_batch_query(
    const float3x3* h_tris, int nA,
    const LBVH& lbvh_B,
    const float3x3* d_tris_B,
    TriQueryResult* h_out);

// ── Plain C++ interface (no CUDA types) ──────────────────────────────────────
// pts: flat array of 3*n_pts doubles [x0,y0,z0, x1,y1,z1, ...]
// out_nearest: output array of n_pts ints (nearest triangle index on B)
// mesh_B_verts: flat array of 9*nB doubles (triangle vertices, same layout as float3x3)
// Must call gpu_plain_init_B() once before repeated gpu_plain_query() calls.

void gpu_plain_init_B(const double* mesh_B_verts, int nB);
void gpu_plain_free_B();
void gpu_plain_query(const double* pts, int n_pts, int* out_nearest);
