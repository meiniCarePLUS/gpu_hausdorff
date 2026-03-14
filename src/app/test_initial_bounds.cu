// Phase-2 test: GPU initial bounds vs CPU traverse() result.
// Usage: test_initial_bounds <meshA.obj> <meshB.obj>
//
// Loads both meshes, builds LBVH for B, runs GPU initial bound estimation,
// then compares against a brute-force CPU reference (point-to-nearest-triangle
// for every vertex of A against all triangles of B).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <vector>

#include <cuda_runtime.h>

#include "bvh/lbvh.cuh"
#include "hausdorff/gpu_initial_bounds.cuh"
#include "core/geometry/gpu_primitive_dis.cuh"

// ── Minimal OBJ loader ────────────────────────────────────────────────────────
static bool load_obj(const char* path,
                     std::vector<float>& verts,
                     std::vector<int>&   tris)
{
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0]=='v' && line[1]==' ') {
            float x,y,z; sscanf(line+2,"%f %f %f",&x,&y,&z);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0]=='f' && line[1]==' ') {
            int a,b,c;
            if (sscanf(line+2,"%d/%*d/%*d %d/%*d/%*d %d/%*d/%*d",&a,&b,&c)==3 ||
                sscanf(line+2,"%d//%*d %d//%*d %d//%*d",&a,&b,&c)==3 ||
                sscanf(line+2,"%d/%*d %d/%*d %d/%*d",&a,&b,&c)==3 ||
                sscanf(line+2,"%d %d %d",&a,&b,&c)==3)
                tris.push_back(a-1), tris.push_back(b-1), tris.push_back(c-1);
        }
    }
    fclose(f);
    return !tris.empty();
}

static std::vector<float3x3> make_tris(const std::vector<float>& v,
                                        const std::vector<int>&   t)
{
    int n = (int)(t.size()/3);
    std::vector<float3x3> out(n);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            int vi = t[i*3+k];
            out[i].v[k*3+0] = v[vi*3+0];
            out[i].v[k*3+1] = v[vi*3+1];
            out[i].v[k*3+2] = v[vi*3+2];
        }
    return out;
}

// ── CPU brute-force reference ─────────────────────────────────────────────────
// For each triangle in A, compute max vertex distance to nearest triangle in B.
// Returns global max (= upper bound = lower bound in this simplified estimate).
static double cpu_reference(const std::vector<float3x3>& tA,
                             const std::vector<float3x3>& tB)
{
    double global_max = 0;
    for (auto& ta : tA) {
        double tri_max = 0;
        for (int k = 0; k < 3; ++k) {
            d3 p = {(double)ta.v[k*3], (double)ta.v[k*3+1], (double)ta.v[k*3+2]};
            double best = DBL_MAX;
            for (auto& tb : tB) {
                double dv[9];
                for (int j = 0; j < 9; ++j) dv[j] = (double)tb.v[j];
                d3 cp;
                double d = pt_tri_sqr_dis(p, dv, &cp);
                if (d < best) best = d;
            }
            if (best > tri_max) tri_max = best;
        }
        if (tri_max > global_max) global_max = tri_max;
    }
    return global_max;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <meshA.obj> <meshB.obj>\n", argv[0]);
        return 1;
    }

    std::vector<float> vA, vB;
    std::vector<int>   tA, tB;
    if (!load_obj(argv[1], vA, tA) || !load_obj(argv[2], vB, tB)) return 1;

    auto hA = make_tris(vA, tA);
    auto hB = make_tris(vB, tB);
    int nA = (int)hA.size(), nB = (int)hB.size();
    printf("Mesh A: %d tris   Mesh B: %d tris\n", nA, nB);

    // ── Build LBVH for B ──────────────────────────────────────────────────
    LBVH lbvh{};
    lbvh.build(hB.data(), nB);

    // ── Upload A to device ────────────────────────────────────────────────
    float3x3 *d_tA, *d_tB;
    cudaMalloc(&d_tA, nA * sizeof(float3x3));
    cudaMalloc(&d_tB, nB * sizeof(float3x3));
    cudaMemcpy(d_tA, hA.data(), nA * sizeof(float3x3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tB, hB.data(), nB * sizeof(float3x3), cudaMemcpyHostToDevice);

    InitialBounds* d_out;
    cudaMalloc(&d_out, sizeof(InitialBounds));

    // ── GPU timed run ─────────────────────────────────────────────────────
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gpu_compute_initial_bounds(d_tA, nA, d_tB, nB, lbvh, d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, t0, t1);

    InitialBounds h_out;
    cudaMemcpy(&h_out, d_out, sizeof(InitialBounds), cudaMemcpyDeviceToHost);
    printf("GPU initial bounds: L=%.6f  U=%.6f  (%.3f ms)\n",
           sqrt(h_out.L), sqrt(h_out.U), gpu_ms);

    // ── CPU reference ─────────────────────────────────────────────────────
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    double cpu_U = cpu_reference(hA, hB);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
    printf("CPU brute-force:    U=%.6f  (%.3f ms)\n", sqrt(cpu_U), cpu_ms);

    // ── Validate ──────────────────────────────────────────────────────────
    double rel_err = fabs(sqrt(h_out.U) - sqrt(cpu_U)) / (sqrt(cpu_U) + 1e-15);
    printf("Relative error: %.2e  %s\n", rel_err,
           rel_err < 1e-3 ? "PASS" : "FAIL (check LBVH traversal)");
    printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);

    cudaFree(d_tA); cudaFree(d_tB); cudaFree(d_out);
    lbvh.free();
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return rel_err >= 1e-3;
}
