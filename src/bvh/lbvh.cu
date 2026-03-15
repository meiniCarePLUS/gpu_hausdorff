#include "lbvh.cuh"

#include <algorithm>
#include <stdexcept>
#include <vector>

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

// CPU recursive build: internal nodes [0..n-2], leaves [n-1..2n-2].
static int build_cpu(std::vector<LBVHNode>& nodes,
                     const std::vector<std::pair<uint32_t,int>>& sorted,
                     const float3x3* tris,
                     int lo, int hi, int& ni, int& nl)
{
    if (lo == hi) {
        int idx = nl++;
        int pi  = sorted[lo].second;
        nodes[idx].prim_idx = pi;
        nodes[idx].left = nodes[idx].right = -1;
        for (int k = 0; k < 3; ++k) {
            float a=tris[pi].v[k], b=tris[pi].v[k+3], c=tris[pi].v[k+6];
            nodes[idx].aabb_min[k] = fminf(fminf(a,b),c);
            nodes[idx].aabb_max[k] = fmaxf(fmaxf(a,b),c);
        }
        return idx;
    }
    int idx = ni++;
    nodes[idx].prim_idx = -1;
    int mid = (lo + hi) / 2;
    int lc  = build_cpu(nodes, sorted, tris, lo,    mid, ni, nl);
    int rc  = build_cpu(nodes, sorted, tris, mid+1, hi,  ni, nl);
    nodes[idx].left  = lc;
    nodes[idx].right = rc;
    nodes[lc].parent = nodes[rc].parent = idx;
    for (int k = 0; k < 3; ++k) {
        nodes[idx].aabb_min[k] = fminf(nodes[lc].aabb_min[k], nodes[rc].aabb_min[k]);
        nodes[idx].aabb_max[k] = fmaxf(nodes[lc].aabb_max[k], nodes[rc].aabb_max[k]);
    }
    return idx;
}

void LBVH::build(const float3x3* h_tris, int n) {
    if (n <= 0) throw std::invalid_argument("LBVH::build: n must be > 0");
    n_prims = n;

    float smin[3]={1e30f,1e30f,1e30f}, smax[3]={-1e30f,-1e30f,-1e30f};
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            float a=h_tris[i].v[k], b=h_tris[i].v[k+3], c=h_tris[i].v[k+6];
            smin[k]=fminf(smin[k],fminf(fminf(a,b),c));
            smax[k]=fmaxf(smax[k],fmaxf(fmaxf(a,b),c));
        }
    float inv[3];
    for (int k = 0; k < 3; ++k) inv[k]=1.f/fmaxf(smax[k]-smin[k],1e-10f);

    std::vector<std::pair<uint32_t,int>> sorted(n);
    for (int i = 0; i < n; ++i) {
        float cx=(h_tris[i].v[0]+h_tris[i].v[3]+h_tris[i].v[6])*(1.f/3.f);
        float cy=(h_tris[i].v[1]+h_tris[i].v[4]+h_tris[i].v[7])*(1.f/3.f);
        float cz=(h_tris[i].v[2]+h_tris[i].v[5]+h_tris[i].v[8])*(1.f/3.f);
        sorted[i]={morton3D_h((cx-smin[0])*inv[0],(cy-smin[1])*inv[1],(cz-smin[2])*inv[2]),i};
    }
    std::stable_sort(sorted.begin(), sorted.end());

    std::vector<LBVHNode> h_nodes(2*n-1);
    h_nodes[0].parent = -1;
    int ni=0, nl=n-1;
    build_cpu(h_nodes, sorted, h_tris, 0, n-1, ni, nl);

    std::vector<int>      h_idx(n);
    std::vector<uint32_t> h_morton(n);
    for (int i = 0; i < n; ++i) { h_idx[i]=sorted[i].second; h_morton[i]=sorted[i].first; }

    cudaMalloc(&d_nodes,      (2*n-1)*sizeof(LBVHNode));
    cudaMalloc(&d_sorted_idx, n*sizeof(int));
    cudaMalloc(&d_morton,     n*sizeof(uint32_t));
    cudaMemcpy(d_nodes,      h_nodes.data(),  (2*n-1)*sizeof(LBVHNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_idx, h_idx.data(),    n*sizeof(int),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_morton,     h_morton.data(), n*sizeof(uint32_t),       cudaMemcpyHostToDevice);
}

void LBVH::free() {
    cudaFree(d_nodes);      d_nodes      = nullptr;
    cudaFree(d_sorted_idx); d_sorted_idx = nullptr;
    cudaFree(d_morton);     d_morton     = nullptr;
}
