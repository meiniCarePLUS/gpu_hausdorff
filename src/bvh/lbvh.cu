#include "lbvh.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

// ─── Morton code helpers ─────────────────────────────────────────────────────

static inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.f, 0.f), 1023.f);
    y = fminf(fmaxf(y * 1024.f, 0.f), 1023.f);
    z = fminf(fmaxf(z * 1024.f, 0.f), 1023.f);
    return (expand_bits((uint32_t)x) << 2) |
           (expand_bits((uint32_t)y) << 1) |
            expand_bits((uint32_t)z);
}

// ─── CPU BVH build (recursive, SAH-free median split) ────────────────────────
// Builds into a flat array of LBVHNode on the host, then uploads to device.
// Node layout: internal nodes [0..n-2], leaf nodes [n-1..2n-2].
// This matches the layout expected by lbvh_nearest_sqr.

struct BuildTri {
    uint32_t morton;
    int      orig_idx;
};

static void aabb_of_tri(const float3x3& tri, float* bmin, float* bmax) {
    for (int k = 0; k < 3; ++k) {
        float a = tri.v[k], b = tri.v[k+3], c = tri.v[k+6];
        bmin[k] = fminf(fminf(a,b),c);
        bmax[k] = fmaxf(fmaxf(a,b),c);
    }
}

// Recursively build subtree covering sorted_tris[lo..hi].
// Returns the node index assigned to this subtree.
// next_internal: pointer to next available internal node index (starts at 0).
// next_leaf:     pointer to next available leaf node index (starts at n-1).
static int build_recursive(
    std::vector<LBVHNode>& nodes,
    const std::vector<BuildTri>& sorted_tris,
    const float3x3* tris,
    int lo, int hi,
    int& next_internal, int& next_leaf)
{
    if (lo == hi) {
        // Leaf
        int idx = next_leaf++;
        nodes[idx].prim_idx = sorted_tris[lo].orig_idx;
        nodes[idx].left     = -1;
        nodes[idx].right    = -1;
        aabb_of_tri(tris[sorted_tris[lo].orig_idx],
                    nodes[idx].aabb_min, nodes[idx].aabb_max);
        return idx;
    }

    // Internal node
    int idx = next_internal++;
    nodes[idx].prim_idx = -1;

    // Split at median
    int mid = (lo + hi) / 2;
    int lc = build_recursive(nodes, sorted_tris, tris, lo,    mid, next_internal, next_leaf);
    int rc = build_recursive(nodes, sorted_tris, tris, mid+1, hi,  next_internal, next_leaf);

    nodes[idx].left  = lc;
    nodes[idx].right = rc;
    nodes[lc].parent = idx;
    nodes[rc].parent = idx;

    // Merge AABBs
    for (int k = 0; k < 3; ++k) {
        nodes[idx].aabb_min[k] = fminf(nodes[lc].aabb_min[k], nodes[rc].aabb_min[k]);
        nodes[idx].aabb_max[k] = fmaxf(nodes[lc].aabb_max[k], nodes[rc].aabb_max[k]);
    }
    return idx;
}

// ─── Host: LBVH::build ───────────────────────────────────────────────────────

void LBVH::build(const float3x3* h_tris, int n) {
    if (n <= 0) throw std::invalid_argument("LBVH::build: n must be > 0");
    n_prims = n;

    // Compute scene AABB
    float smin[3] = {1e30f, 1e30f, 1e30f};
    float smax[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            float a = h_tris[i].v[k], b = h_tris[i].v[k+3], c = h_tris[i].v[k+6];
            smin[k] = fminf(smin[k], fminf(fminf(a,b),c));
            smax[k] = fmaxf(smax[k], fmaxf(fmaxf(a,b),c));
        }
    float inv[3];
    for (int k = 0; k < 3; ++k)
        inv[k] = 1.f / fmaxf(smax[k]-smin[k], 1e-10f);

    // Compute Morton codes and sort
    std::vector<BuildTri> bt(n);
    for (int i = 0; i < n; ++i) {
        float cx = (h_tris[i].v[0]+h_tris[i].v[3]+h_tris[i].v[6])*(1.f/3.f);
        float cy = (h_tris[i].v[1]+h_tris[i].v[4]+h_tris[i].v[7])*(1.f/3.f);
        float cz = (h_tris[i].v[2]+h_tris[i].v[5]+h_tris[i].v[8])*(1.f/3.f);
        bt[i].morton   = morton3D((cx-smin[0])*inv[0], (cy-smin[1])*inv[1], (cz-smin[2])*inv[2]);
        bt[i].orig_idx = i;
    }
    std::stable_sort(bt.begin(), bt.end(),
        [](const BuildTri& a, const BuildTri& b){ return a.morton < b.morton; });

    // Build tree on CPU
    // Node layout: internal [0..n-2], leaf [n-1..2n-2]
    std::vector<LBVHNode> h_nodes(2*n-1);
    // Init parent of root to -1
    h_nodes[0].parent = -1;

    int ni = 0, nl = n-1;
    int root = build_recursive(h_nodes, bt, h_tris, 0, n-1, ni, nl);
    // root must be node 0 for traversal starting at index 0
    // build_recursive assigns internal nodes in pre-order, so root = 0. ✓

    // Upload to device
    cudaMalloc(&d_nodes,      (2*n-1) * sizeof(LBVHNode));
    cudaMalloc(&d_sorted_idx, n * sizeof(int));
    cudaMalloc(&d_morton,     n * sizeof(uint32_t));

    cudaMemcpy(d_nodes, h_nodes.data(), (2*n-1)*sizeof(LBVHNode), cudaMemcpyHostToDevice);

    // sorted_idx and morton for reference (not used by traversal)
    std::vector<int>      h_idx(n);
    std::vector<uint32_t> h_morton(n);
    for (int i = 0; i < n; ++i) { h_idx[i] = bt[i].orig_idx; h_morton[i] = bt[i].morton; }
    cudaMemcpy(d_sorted_idx, h_idx.data(),    n*sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_morton,     h_morton.data(), n*sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void LBVH::free() {
    cudaFree(d_nodes);      d_nodes      = nullptr;
    cudaFree(d_sorted_idx); d_sorted_idx = nullptr;
    cudaFree(d_morton);     d_morton     = nullptr;
}
