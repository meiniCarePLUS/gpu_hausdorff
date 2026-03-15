#include "gpu_batch_query.cuh"
#include "core/geometry/gpu_primitive_dis.cuh"

#include <float.h>

// Reuse the same LBVH traversal from gpu_initial_bounds.cu
__device__ static double lbvh_nearest_sqr_q(
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
            double d = 0, px[3] = {p.x, p.y, p.z};
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

__global__ static void kernel_batch_query(
    const float3x3* __restrict__ tris_A, int nA,
    const float3x3* __restrict__ tris_B,
    const LBVHNode* __restrict__ nodes_B,
    TriQueryResult* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nA) return;

    const float* fv = tris_A[i].v;
    for (int k = 0; k < 3; ++k) {
        d3 p = {(double)fv[k*3], (double)fv[k*3+1], (double)fv[k*3+2]};
        out[i].d[k] = lbvh_nearest_sqr_q(p, nodes_B, tris_B);
    }
}

void gpu_batch_query(
    const float3x3* h_tris, int nA,
    const LBVH& lbvh_B,
    const float3x3* d_tris_B,
    TriQueryResult* h_out)
{
    float3x3* d_tris_A;
    TriQueryResult* d_out;
    cudaMalloc(&d_tris_A, nA * sizeof(float3x3));
    cudaMalloc(&d_out,    nA * sizeof(TriQueryResult));
    cudaMemcpy(d_tris_A, h_tris, nA * sizeof(float3x3), cudaMemcpyHostToDevice);

    int block = 128, grid = (nA + block - 1) / block;
    kernel_batch_query<<<grid, block>>>(d_tris_A, nA, d_tris_B, lbvh_B.d_nodes, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, nA * sizeof(TriQueryResult), cudaMemcpyDeviceToHost);
    cudaFree(d_tris_A);
    cudaFree(d_out);
}
