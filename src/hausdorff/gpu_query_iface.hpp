#pragma once
// Plain C++ interface to GPU nearest-triangle query.
// No CUDA types exposed — safe to include from .cpp files.

// Initialize GPU state for mesh B. Call once before any gpu_plain_query().
// mesh_B_verts: flat array of 9*nB doubles, layout [v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z] per triangle.
void gpu_plain_init_B(const double* mesh_B_verts, int nB);

// Release GPU resources for mesh B.
void gpu_plain_free_B();

// Query nearest triangle on B for n_pts points.
// pts: flat array of 3*n_pts doubles [x0,y0,z0, x1,y1,z1, ...]
// out_nearest: output array of n_pts ints (0-based triangle index on B)
void gpu_plain_query(const double* pts, int n_pts, int* out_nearest);
