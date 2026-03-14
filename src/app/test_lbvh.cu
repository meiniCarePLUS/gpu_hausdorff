// Phase-1 benchmark: GPU LBVH build vs CPU BVH build.
// Usage: test_lbvh <mesh.obj>
//   Loads the mesh, converts to float3x3 triangles, builds LBVH on GPU,
//   prints build time and validates that every leaf AABB contains its triangle.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "bvh/lbvh.cuh"

// ── Minimal OBJ loader (vertices + triangles only) ───────────────────────────
static bool load_obj(const char* path,
                     std::vector<float>& verts,   // flat xyz
                     std::vector<int>&   tris)    // flat v0v1v2
{
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            sscanf(line + 2, "%f %f %f", &x, &y, &z);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            int a, b, c;
            // handle "f v", "f v/vt", "f v/vt/vn", "f v//vn"
            if (sscanf(line + 2, "%d/%*d/%*d %d/%*d/%*d %d/%*d/%*d", &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d//%*d %d//%*d %d//%*d",           &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d/%*d %d/%*d %d/%*d",              &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d %d %d",                          &a, &b, &c) == 3) {
                tris.push_back(a-1); tris.push_back(b-1); tris.push_back(c-1);
            }
        }
    }
    fclose(f);
    return !tris.empty();
}

// ── Validation kernel: check each leaf AABB contains its triangle ─────────────
__global__ void kernel_validate(
    const float3x3* __restrict__ tris,
    const LBVHNode* __restrict__ nodes,
    int n, int* __restrict__ errors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int leaf = i + n - 1;
    int pi   = nodes[leaf].prim_idx;
    if (pi < 0) { atomicAdd(errors, 1); return; }

    const float* v = tris[pi].v;
    for (int k = 0; k < 3; ++k) {
        float a = v[k], b = v[k+3], c = v[k+6];
        float lo = fminf(fminf(a,b),c), hi = fmaxf(fmaxf(a,b),c);
        if (lo < nodes[leaf].aabb_min[k] - 1e-5f ||
            hi > nodes[leaf].aabb_max[k] + 1e-5f)
            atomicAdd(errors, 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mesh.obj>\n", argv[0]);
        return 1;
    }

    // ── Load mesh ──────────────────────────────────────────────────────────
    std::vector<float> verts;
    std::vector<int>   tris_idx;
    if (!load_obj(argv[1], verts, tris_idx)) return 1;

    int nv = (int)(verts.size() / 3);
    int nt = (int)(tris_idx.size() / 3);
    printf("Mesh: %d vertices, %d triangles\n", nv, nt);

    // ── Build float3x3 array ───────────────────────────────────────────────
    std::vector<float3x3> h_tris(nt);
    for (int i = 0; i < nt; ++i) {
        for (int k = 0; k < 3; ++k) {
            int vi = tris_idx[i*3 + k];
            h_tris[i].v[k*3 + 0] = verts[vi*3 + 0];
            h_tris[i].v[k*3 + 1] = verts[vi*3 + 1];
            h_tris[i].v[k*3 + 2] = verts[vi*3 + 2];
        }
    }

    // ── GPU LBVH build (timed) ─────────────────────────────────────────────
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    LBVH lbvh{};
    cudaEventRecord(t0);
    lbvh.build(h_tris.data(), nt);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("GPU LBVH build: %.3f ms  (%d nodes)\n", ms, 2*nt - 1);

    // ── Validate leaf AABBs ────────────────────────────────────────────────
    float3x3* d_tris;
    cudaMalloc(&d_tris, nt * sizeof(float3x3));
    cudaMemcpy(d_tris, h_tris.data(), nt * sizeof(float3x3), cudaMemcpyHostToDevice);

    int* d_errors;
    cudaMalloc(&d_errors, sizeof(int));
    cudaMemset(d_errors, 0, sizeof(int));

    int block = 256, grid = (nt + block - 1) / block;
    kernel_validate<<<grid, block>>>(d_tris, lbvh.d_nodes, nt, d_errors);

    int h_errors = 0;
    cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Validation: %s (%d leaf AABB errors)\n",
           h_errors == 0 ? "PASS" : "FAIL", h_errors);

    cudaFree(d_tris);
    cudaFree(d_errors);
    lbvh.free();
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return h_errors != 0;
}
