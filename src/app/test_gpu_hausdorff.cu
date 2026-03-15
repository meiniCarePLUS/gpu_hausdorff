// Phase-3 test: GPU batch nearest-triangle query vs CPU BVH.
// Usage: test_gpu_hausdorff <meshA.obj> <meshB.obj>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <vector>

#include <cuda_runtime.h>
#include "bvh/lbvh.cuh"
#include "hausdorff/gpu_batch_query.cuh"
#include "core/geometry/gpu_primitive_dis.cuh"

static bool load_obj(const char* path, std::vector<float>& verts, std::vector<int>& tris) {
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

static std::vector<float3x3> make_tris(const std::vector<float>& v, const std::vector<int>& t) {
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

// CPU brute-force: for each vertex of each triangle in A, find nearest triangle in B.
static std::vector<TriQueryResult> cpu_reference(
    const std::vector<float3x3>& tA, const std::vector<float3x3>& tB)
{
    int nA = (int)tA.size(), nB = (int)tB.size();
    std::vector<TriQueryResult> out(nA);
    for (int i = 0; i < nA; ++i)
        for (int k = 0; k < 3; ++k) {
            d3 p = {(double)tA[i].v[k*3], (double)tA[i].v[k*3+1], (double)tA[i].v[k*3+2]};
            double best = DBL_MAX;
            for (int j = 0; j < nB; ++j) {
                double dv[9]; for (int m = 0; m < 9; ++m) dv[m] = (double)tB[j].v[m];
                d3 cp; double d = pt_tri_sqr_dis(p, dv, &cp);
                if (d < best) best = d;
            }
            out[i].d[k] = best;
        }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <meshA.obj> <meshB.obj>\n", argv[0]); return 1; }

    std::vector<float> vA, vB; std::vector<int> tA, tB;
    if (!load_obj(argv[1], vA, tA) || !load_obj(argv[2], vB, tB)) return 1;

    auto hA = make_tris(vA, tA), hB = make_tris(vB, tB);
    int nA = (int)hA.size(), nB = (int)hB.size();
    printf("Mesh A: %d tris   Mesh B: %d tris\n", nA, nB);

    // Build LBVH for B and upload B triangles
    LBVH lbvh{};
    lbvh.build(hB.data(), nB);

    float3x3* d_tB;
    cudaMalloc(&d_tB, nB * sizeof(float3x3));
    cudaMemcpy(d_tB, hB.data(), nB * sizeof(float3x3), cudaMemcpyHostToDevice);

    // GPU batch query
    std::vector<TriQueryResult> gpu_out(nA);
    auto g0 = std::chrono::high_resolution_clock::now();
    gpu_batch_query(hA.data(), nA, lbvh, d_tB, gpu_out.data());
    auto g1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double,std::milli>(g1-g0).count();

    // CPU reference
    auto c0 = std::chrono::high_resolution_clock::now();
    auto cpu_out = cpu_reference(hA, hB);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(c1-c0).count();

    // Validate: max relative error across all vertex distances
    double max_rel = 0;
    for (int i = 0; i < nA; ++i)
        for (int k = 0; k < 3; ++k) {
            double ref = sqrt(cpu_out[i].d[k]), got = sqrt(gpu_out[i].d[k]);
            double rel = fabs(ref - got) / (ref + 1e-15);
            if (rel > max_rel) max_rel = rel;
        }

    printf("GPU batch query: %.3f ms\n", gpu_ms);
    printf("CPU brute-force: %.3f ms\n", cpu_ms);
    printf("Max relative error: %.2e  %s\n", max_rel, max_rel < 1e-3 ? "PASS" : "FAIL");
    printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);

    cudaFree(d_tB);
    lbvh.free();
    return max_rel >= 1e-3;
}
