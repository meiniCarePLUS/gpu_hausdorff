#include "gpu_batch_query.cuh"
#include "core/geometry/gpu_primitive_dis.cuh"

#include <float.h>
#include <vector>

__device__ static double lbvh_nearest_sqr_idx(
    d3 p,
    const LBVHNode* __restrict__ nodes,
    const float3x3* __restrict__ tris_B,
    int* out_idx)
{
    int stack[64], top = 0;
    stack[top++] = 0;
    double best = DBL_MAX;
    *out_idx = -1;

    while (top > 0) {
        int ni = stack[--top];
        const LBVHNode& node = nodes[ni];

        if (node.prim_idx >= 0) {
            const float* fv = tris_B[node.prim_idx].v;
            double dv[9];
            for (int k = 0; k < 9; ++k) dv[k] = (double)fv[k];
            d3 cp;
            double d = pt_tri_sqr_dis(p, dv, &cp);
            if (d < best) { best = d; *out_idx = node.prim_idx; }
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
        out[i].d[k] = lbvh_nearest_sqr_idx(p, nodes_B, tris_B, &out[i].nearest[k]);
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

// ── Plain C++ interface ───────────────────────────────────────────────────────

static LBVH      g_lbvh{};
static float3x3* g_d_tris_B  = nullptr;
static int       g_nB         = 0;

// Pre-allocated device buffers for small per-iteration queries (≤ 64 points).
static const int  QUERY_BUF   = 64;
static double*    g_d_pts      = nullptr;
static int*       g_d_nearest  = nullptr;

__global__ static void kernel_point_query(
    const double* __restrict__ pts, int n_pts,
    const float3x3* __restrict__ tris_B,
    const LBVHNode* __restrict__ nodes_B,
    int* __restrict__ out_nearest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pts) return;
    d3 p = {pts[i*3], pts[i*3+1], pts[i*3+2]};
    int idx;
    lbvh_nearest_sqr_idx(p, nodes_B, tris_B, &idx);
    out_nearest[i] = idx;
}

void gpu_plain_init_B(const double* mesh_B_verts, int nB) {
    g_nB = nB;
    std::vector<float3x3> h_tris(nB);
    for (int i = 0; i < nB; ++i)
        for (int k = 0; k < 9; ++k)
            h_tris[i].v[k] = (float)mesh_B_verts[i*9 + k];

    g_lbvh.build(h_tris.data(), nB);
    cudaMalloc(&g_d_tris_B, nB * sizeof(float3x3));
    cudaMemcpy(g_d_tris_B, h_tris.data(), nB * sizeof(float3x3), cudaMemcpyHostToDevice);

    cudaMalloc(&g_d_pts,     QUERY_BUF * 3 * sizeof(double));
    cudaMalloc(&g_d_nearest, QUERY_BUF * sizeof(int));
}

void gpu_plain_free_B() {
    g_lbvh.free();
    cudaFree(g_d_tris_B); g_d_tris_B = nullptr;
    cudaFree(g_d_pts);    g_d_pts    = nullptr;
    cudaFree(g_d_nearest);g_d_nearest= nullptr;
    g_nB = 0;
}

void gpu_plain_query(const double* pts, int n_pts, int* out_nearest) {
    // For very small batches (≤ QUERY_BUF), use pre-allocated buffers.
    // For larger batches, fall back to dynamic allocation.
    double* d_pts;
    int*    d_out;
    bool    dynamic = (n_pts > QUERY_BUF);
    if (dynamic) {
        cudaMalloc(&d_pts, n_pts * 3 * sizeof(double));
        cudaMalloc(&d_out, n_pts * sizeof(int));
    } else {
        d_pts = g_d_pts;
        d_out = g_d_nearest;
    }

    cudaMemcpy(d_pts, pts, n_pts * 3 * sizeof(double), cudaMemcpyHostToDevice);
    int block = 32, grid = (n_pts + block - 1) / block;
    kernel_point_query<<<grid, block>>>(d_pts, n_pts, g_d_tris_B, g_lbvh.d_nodes, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(out_nearest, d_out, n_pts * sizeof(int), cudaMemcpyDeviceToHost);

    if (dynamic) { cudaFree(d_pts); cudaFree(d_out); }
}
