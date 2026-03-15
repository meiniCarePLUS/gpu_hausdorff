#include "lbvh.cuh"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

// ─── Morton code helpers ─────────────────────────────────────────────────────

static inline uint32_t expand_bits_h(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline uint32_t morton3D_h(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.f, 0.f), 1023.f);
    y = fminf(fmaxf(y * 1024.f, 0.f), 1023.f);
    z = fminf(fmaxf(z * 1024.f, 0.f), 1023.f);
    return (expand_bits_h((uint32_t)x) << 2) |
           (expand_bits_h((uint32_t)y) << 1) |
            expand_bits_h((uint32_t)z);
}

// ─── Kernel: binary radix tree (Karras 2012) ─────────────────────────────────

__device__ __forceinline__ int delta(
    const uint32_t* __restrict__ m, int i, int j, int n)
{
    if (j < 0 || j >= n) return -1;
    if (m[i] == m[j])
        return 32 + __clz(i ^ j);
    return __clz(m[i] ^ m[j]);
}

__global__ void kernel_build_tree(
    const uint32_t* __restrict__ sorted_morton,
    LBVHNode* __restrict__ nodes,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    const uint32_t* m = sorted_morton;

    int d = (delta(m, i, i + 1, n) - delta(m, i, i - 1, n)) >= 0 ? 1 : -1;

    int delta_min = delta(m, i, i - d, n);
    int l_max = 2;
    while (delta(m, i, i + l_max * d, n) > delta_min)
        l_max <<= 1;

    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1)
        if (delta(m, i, i + (l + t) * d, n) > delta_min)
            l += t;
    int j = i + l * d;

    int delta_node = delta(m, i, j, n);
    int s = 0;
    int step = l;
    do {
        step = (step + 1) >> 1;
        if (delta(m, i, i + (s + step) * d, n) > delta_node)
            s += step;
    } while (step > 1);
    int gamma = i + s * d + min(d, 0);

    int left_child  = (min(i, j) == gamma)     ? (gamma + n - 1)     : gamma;
    int right_child = (max(i, j) == gamma + 1) ? (gamma + 1 + n - 1) : (gamma + 1);

    nodes[i].left      = left_child;
    nodes[i].right     = right_child;
    nodes[i].prim_idx  = -1;

    nodes[left_child].parent  = i;
    nodes[right_child].parent = i;
}

// ─── Kernel: bottom-up AABB refit ────────────────────────────────────────────

__device__ __forceinline__ void aabb_of_tri(
    const float3x3& tri, float* bmin, float* bmax)
{
    for (int k = 0; k < 3; ++k) {
        float a = tri.v[k], b = tri.v[k + 3], c = tri.v[k + 6];
        bmin[k] = fminf(fminf(a, b), c);
        bmax[k] = fmaxf(fmaxf(a, b), c);
    }
}

__global__ void kernel_refit(
    const float3x3* __restrict__ tris,
    const int* __restrict__ sorted_idx,
    LBVHNode* __restrict__ nodes,
    int* __restrict__ flags,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int leaf = i + n - 1;
    nodes[leaf].prim_idx = sorted_idx[i];
    nodes[leaf].left     = -1;
    nodes[leaf].right    = -1;
    aabb_of_tri(tris[sorted_idx[i]],
                nodes[leaf].aabb_min,
                nodes[leaf].aabb_max);

    int node = nodes[leaf].parent;
    while (node != -1) {
        if (atomicAdd(&flags[node], 1) == 0) return;
        __threadfence();

        int lc = nodes[node].left;
        int rc = nodes[node].right;
        for (int k = 0; k < 3; ++k) {
            nodes[node].aabb_min[k] = fminf(nodes[lc].aabb_min[k],
                                            nodes[rc].aabb_min[k]);
            nodes[node].aabb_max[k] = fmaxf(nodes[lc].aabb_max[k],
                                            nodes[rc].aabb_max[k]);
        }
        node = nodes[node].parent;
    }
}

// ─── Host: LBVH::build ───────────────────────────────────────────────────────

void LBVH::build(const float3x3* h_tris, int n) {
    if (n <= 0) throw std::invalid_argument("LBVH::build: n must be > 0");
    n_prims = n;

    // Compute scene AABB on host.
    float smin[3] = {1e30f, 1e30f, 1e30f};
    float smax[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            float a = h_tris[i].v[k], b = h_tris[i].v[k+3], c = h_tris[i].v[k+6];
            float lo = fminf(fminf(a,b),c), hi = fmaxf(fmaxf(a,b),c);
            smin[k] = fminf(smin[k], lo);
            smax[k] = fmaxf(smax[k], hi);
        }
    float inv[3];
    for (int k = 0; k < 3; ++k)
        inv[k] = 1.f / fmaxf(smax[k]-smin[k], 1e-10f);

    // Compute Morton codes and sort on CPU (fast for moderate n, avoids Thrust overhead).
    struct BT { uint32_t morton; int idx; };
    std::vector<BT> bt(n);
    for (int i = 0; i < n; ++i) {
        float cx = (h_tris[i].v[0]+h_tris[i].v[3]+h_tris[i].v[6])*(1.f/3.f);
        float cy = (h_tris[i].v[1]+h_tris[i].v[4]+h_tris[i].v[7])*(1.f/3.f);
        float cz = (h_tris[i].v[2]+h_tris[i].v[5]+h_tris[i].v[8])*(1.f/3.f);
        bt[i].morton = morton3D_h((cx-smin[0])*inv[0], (cy-smin[1])*inv[1], (cz-smin[2])*inv[2]);
        bt[i].idx    = i;
    }
    std::stable_sort(bt.begin(), bt.end(), [](const BT& a, const BT& b){ return a.morton < b.morton; });

    std::vector<uint32_t> h_morton(n);
    std::vector<int>      h_idx(n);
    for (int i = 0; i < n; ++i) { h_morton[i] = bt[i].morton; h_idx[i] = bt[i].idx; }

    // Upload sorted data and triangles.
    float3x3* d_tris;
    cudaMalloc(&d_tris,       n * sizeof(float3x3));
    cudaMalloc(&d_morton,     n * sizeof(uint32_t));
    cudaMalloc(&d_sorted_idx, n * sizeof(int));
    cudaMalloc(&d_nodes,      (2*n-1) * sizeof(LBVHNode));

    cudaMemcpy(d_tris,       h_tris,          n * sizeof(float3x3),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_morton,     h_morton.data(), n * sizeof(uint32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_idx, h_idx.data(),    n * sizeof(int),       cudaMemcpyHostToDevice);

    int block = 256, grid = (n + block - 1) / block;

    // Step 1: Build binary radix tree on GPU.
    cudaMemset(d_nodes, 0xFF, (2*n-1) * sizeof(LBVHNode));
    if (n > 1) {
        int igrid = (n - 1 + block - 1) / block;
        kernel_build_tree<<<igrid, block>>>(d_morton, d_nodes, n);
    }
    cudaDeviceSynchronize();

    // Step 2: Refit AABBs bottom-up on GPU.
    int* d_flags;
    cudaMalloc(&d_flags, (n-1) * sizeof(int));
    cudaMemset(d_flags, 0, (n-1) * sizeof(int));
    kernel_refit<<<grid, block>>>(d_tris, d_sorted_idx, d_nodes, d_flags, n);
    cudaDeviceSynchronize();

    cudaFree(d_flags);
    cudaFree(d_tris);
}

void LBVH::free() {
    cudaFree(d_nodes);      d_nodes      = nullptr;
    cudaFree(d_sorted_idx); d_sorted_idx = nullptr;
    cudaFree(d_morton);     d_morton     = nullptr;
}
