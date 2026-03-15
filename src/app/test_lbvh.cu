// Phase-1 test: GPU LBVH build validation.
// Usage: test_lbvh <mesh.obj>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "bvh/lbvh.cuh"

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

// Validate leaf AABBs on CPU by downloading nodes.
static int validate(const LBVH& lbvh, const std::vector<float3x3>& h_tris) {
    int n = lbvh.n_prims;
    std::vector<LBVHNode> h_nodes(2*n-1);
    cudaMemcpy(h_nodes.data(), lbvh.d_nodes, (2*n-1)*sizeof(LBVHNode), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = n-1; i < 2*n-1; ++i) {
        int pi = h_nodes[i].prim_idx;
        if (pi < 0) { ++errors; continue; }
        const float* v = h_tris[pi].v;
        for (int k = 0; k < 3; ++k) {
            float a = v[k], b = v[k+3], c = v[k+6];
            float lo = fminf(fminf(a,b),c), hi = fmaxf(fmaxf(a,b),c);
            if (lo < h_nodes[i].aabb_min[k] - 1e-5f ||
                hi > h_nodes[i].aabb_max[k] + 1e-5f) ++errors;
        }
    }
    return errors;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <mesh.obj>\n", argv[0]); return 1; }

    std::vector<float> verts; std::vector<int> tris_idx;
    if (!load_obj(argv[1], verts, tris_idx)) return 1;

    int nt = (int)(tris_idx.size()/3);
    printf("Mesh: %d vertices, %d triangles\n", (int)(verts.size()/3), nt);

    std::vector<float3x3> h_tris(nt);
    for (int i = 0; i < nt; ++i)
        for (int k = 0; k < 3; ++k) {
            int vi = tris_idx[i*3+k];
            h_tris[i].v[k*3+0] = verts[vi*3+0];
            h_tris[i].v[k*3+1] = verts[vi*3+1];
            h_tris[i].v[k*3+2] = verts[vi*3+2];
        }

    auto t0 = std::chrono::high_resolution_clock::now();
    LBVH lbvh{};
    lbvh.build(h_tris.data(), nt);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("LBVH build: %.3f ms  (%d nodes)\n", ms, 2*nt-1);

    int errs = validate(lbvh, h_tris);
    printf("Validation: %s (%d leaf AABB errors)\n", errs==0?"PASS":"FAIL", errs);

    lbvh.free();
    return errs != 0;
}
