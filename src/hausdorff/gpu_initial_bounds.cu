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

__device__ double lbvh_nearest_sqr(
    d3 p,
    const LBVHNode* __restrict__ nodes,
    const float3x3* __restrict__ tris_B)
{
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    double best = DBL_MAX;

    while (top > 0) {
        int node_idx = stack[--top];
        const LBVHNode& node = nodes[node_idx];

        if (node.prim_idx >= 0) {
            const float* fv = tris_B[node.prim_idx].v;
            double dv[9];
            for (int k = 0; k < 9; ++k) dv[k] = (double)fv[k];
            d3 cp;
            double d = pt_tri_sqr_dis(p, dv, &cp);
            if (d < best) best = d;
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
// Each thread handles one triangle A_i.
// local bound = max over 3 vertices of (squared dist to nearest triangle on B).
// Global L = Global U = max over all triangles (tightest valid upper bound).

__global__ void kernel_initial_bounds(
    const float3x3* __restrict__ tris_A, int nA,
    const float3x3* __restrict__ tris_B,
    const LBVHNode* __restrict__ nodes_B,
    InitialBounds* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nA) return;

    const float* fv = tris_A[i].v;
    d3 v0 = {(double)fv[0], (double)fv[1], (double)fv[2]};
    d3 v1 = {(double)fv[3], (double)fv[4], (double)fv[5]};
    d3 v2 = {(double)fv[6], (double)fv[7], (double)fv[8]};

    double d0 = lbvh_nearest_sqr(v0, nodes_B, tris_B);
    double d1 = lbvh_nearest_sqr(v1, nodes_B, tris_B);
    double d2 = lbvh_nearest_sqr(v2, nodes_B, tris_B);

    double local_val = d0;
    if (d1 > local_val) local_val = d1;
    if (d2 > local_val) local_val = d2;

    atomicMaxDouble(&out->L, local_val);
    atomicMaxDouble(&out->U, local_val);
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
        d_out);
    cudaDeviceSynchronize();
}
