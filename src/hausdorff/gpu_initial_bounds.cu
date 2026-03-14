#include "gpu_initial_bounds.cuh"
#include "core/geometry/gpu_primitive_dis.cuh"
#include "bvh/lbvh.cuh"

#include <float.h>

// ─── atomicMax/Min for double via CAS ────────────────────────────────────────

__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) return;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(val));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMinDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val <= val) return;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(val));
    } while (assumed != old);
}

// ─── LBVH nearest-triangle traversal ─────────────────────────────────────────
// Returns squared distance from p to nearest triangle; writes nearest prim idx.

__device__ double lbvh_nearest_sqr(
    d3 p,
    const LBVHNode* __restrict__ nodes,
    const float3x3* __restrict__ tris_B,
    int n,
    int* nearest_idx)
{
    int stack[64];
    int top = 0;
    stack[top++] = 0;

    double best = DBL_MAX;
    *nearest_idx = -1;

    while (top > 0) {
        int node_idx = stack[--top];
        const LBVHNode& node = nodes[node_idx];

        if (node.prim_idx >= 0) {
            const float* fv = tris_B[node.prim_idx].v;
            double dv[9];
            for (int k = 0; k < 9; ++k) dv[k] = (double)fv[k];
            d3 cp;
            double d = pt_tri_sqr_dis(p, dv, &cp);
            if (d < best) { best = d; *nearest_idx = node.prim_idx; }
            continue;
        }

        auto aabb_sqr = [&](int ci) -> double {
            const LBVHNode& ch = nodes[ci];
            double d = 0;
            double px[3] = {p.x, p.y, p.z};
            for (int k = 0; k < 3; ++k) {
                if      (px[k] < ch.aabb_min[k]) { double diff = ch.aabb_min[k]-px[k]; d += diff*diff; }
                else if (px[k] > ch.aabb_max[k]) { double diff = px[k]-ch.aabb_max[k]; d += diff*diff; }
            }
            return d;
        };

        int lc = node.left, rc = node.right;
        double dl = aabb_sqr(lc), dr = aabb_sqr(rc);
        if (dl <= dr) {
            if (dr < best) stack[top++] = rc;
            if (dl < best) stack[top++] = lc;
        } else {
            if (dl < best) stack[top++] = lc;
            if (dr < best) stack[top++] = rc;
        }
    }
    return best;
}

// ─── Main kernel ─────────────────────────────────────────────────────────────
// For each triangle A_i:
//   - Find nearest triangle on B for each of the 3 vertices (+ centroid).
//   - local_L = max vertex distance (lower bound: farthest vertex is hardest to cover).
//   - local_U = min over candidate triangles T_j of max(dist(v0,T_j), dist(v1,T_j), dist(v2,T_j))
//               (upper bound: there exists a triangle on B that covers all vertices within this distance).

__device__ double tri_max_vertex_dis(
    const d3* pts, int npts,
    const float3x3* __restrict__ tris_B, int prim_idx)
{
    const float* fv = tris_B[prim_idx].v;
    double dv[9];
    for (int k = 0; k < 9; ++k) dv[k] = (double)fv[k];
    double mx = 0;
    for (int k = 0; k < npts; ++k) {
        d3 cp;
        double d = pt_tri_sqr_dis(pts[k], dv, &cp);
        if (d > mx) mx = d;
    }
    return mx;
}

__global__ void kernel_initial_bounds(
    const float3x3* __restrict__ tris_A, int nA,
    const float3x3* __restrict__ tris_B,
    const LBVHNode* __restrict__ nodes_B,
    int nB,
    InitialBounds* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nA) return;

    const float* fv = tris_A[i].v;
    d3 pts[4];
    pts[0] = {(double)fv[0], (double)fv[1], (double)fv[2]};
    pts[1] = {(double)fv[3], (double)fv[4], (double)fv[5]};
    pts[2] = {(double)fv[6], (double)fv[7], (double)fv[8]};
    pts[3] = {(pts[0].x+pts[1].x+pts[2].x)/3.0,
              (pts[0].y+pts[1].y+pts[2].y)/3.0,
              (pts[0].z+pts[1].z+pts[2].z)/3.0};

    // Find nearest triangle on B for each vertex + centroid.
    int nearest[4];
    double d[4];
    for (int k = 0; k < 4; ++k)
        d[k] = lbvh_nearest_sqr(pts[k], nodes_B, tris_B, nB, &nearest[k]);

    // local_L: max vertex distance (lower bound contribution).
    double local_L = d[0];
    if (d[1] > local_L) local_L = d[1];
    if (d[2] > local_L) local_L = d[2];

    // local_U: for each candidate triangle (from nearest[0..3]), compute
    // max distance from all 3 vertices to that triangle; take the minimum.
    // This matches CPU iterate_leaf(): find a single triangle on B that
    // covers all vertices of A_i as tightly as possible.
    double local_U = DBL_MAX;
    // deduplicate candidates (up to 4, often overlapping)
    int seen[4] = {-1,-1,-1,-1};
    for (int k = 0; k < 4; ++k) {
        int idx = nearest[k];
        if (idx < 0) continue;
        bool dup = false;
        for (int j = 0; j < k; ++j) if (seen[j] == idx) { dup = true; break; }
        if (dup) continue;
        seen[k] = idx;
        // max distance from 3 vertices (not centroid) to this candidate
        double mx = tri_max_vertex_dis(pts, 3, tris_B, idx);
        if (mx < local_U) local_U = mx;
    }

    atomicMaxDouble(&out->L, local_L);
    atomicMaxDouble(&out->U, local_U);
}

// ─── Host entry point ─────────────────────────────────────────────────────────

void gpu_compute_initial_bounds(
    const float3x3* d_tris_A, int nA,
    const float3x3* d_tris_B, int nB,
    const LBVH& lbvh_B,
    InitialBounds* d_out)
{
    // Initialise output: L=0, U=0 (will be updated to max values)
    InitialBounds zero{0.0, 0.0};
    cudaMemcpy(d_out, &zero, sizeof(InitialBounds), cudaMemcpyHostToDevice);

    int block = 128, grid = (nA + block - 1) / block;
    kernel_initial_bounds<<<grid, block>>>(
        d_tris_A, nA,
        d_tris_B,
        lbvh_B.d_nodes,
        nB,
        d_out);
    cudaDeviceSynchronize();
}
