#pragma once
#include <cuda_runtime.h>
#include <float.h>

// ─── Minimal 3-vector ops (double precision) ─────────────────────────────────

struct d3 {
    double x, y, z;
};

__host__ __device__ __forceinline__ d3 d3sub(d3 a, d3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ __forceinline__ double d3dot(d3 a, d3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ __forceinline__ d3 d3scale(d3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
__host__ __device__ __forceinline__ d3 d3add(d3 a, d3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

// ─── Point-to-triangle squared distance (Ericson RTCD §5.1.5) ────────────────
// tri: 9 doubles, column-major: [p0x,p0y,p0z, p1x,p1y,p1z, p2x,p2y,p2z]
// Returns squared distance; writes closest point to *cp.
__host__ __device__ inline double pt_tri_sqr_dis(d3 p, const double *__restrict__ tri, d3 *cp) {
    d3 a = {tri[0], tri[1], tri[2]};
    d3 b = {tri[3], tri[4], tri[5]};
    d3 c = {tri[6], tri[7], tri[8]};

    d3 ab = d3sub(b, a), ac = d3sub(c, a), ap = d3sub(p, a);
    double d1 = d3dot(ab, ap), d2 = d3dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        *cp = a;
        return d3dot(ap, ap);
    }

    d3 bp = d3sub(p, b);
    double d3_ = d3dot(ab, bp), d4 = d3dot(ac, bp);
    if (d3_ >= 0.0 && d4 <= d3_) {
        *cp = b;
        d3 v = d3sub(p, b);
        return d3dot(v, v);
    }

    double vc = d1 * d4 - d3_ * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3_ <= 0.0) {
        double v = d1 / (d1 - d3_);
        *cp = d3add(a, d3scale(ab, v));
        d3 v2 = d3sub(p, *cp);
        return d3dot(v2, v2);
    }

    d3 cp2 = d3sub(p, c);
    double d5 = d3dot(ab, cp2), d6 = d3dot(ac, cp2);
    if (d6 >= 0.0 && d5 <= d6) {
        *cp = c;
        d3 v = d3sub(p, c);
        return d3dot(v, v);
    }

    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double w = d2 / (d2 - d6);
        *cp = d3add(a, d3scale(ac, w));
        d3 v = d3sub(p, *cp);
        return d3dot(v, v);
    }

    double va = d3_ * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3_) >= 0.0 && (d5 - d6) >= 0.0) {
        double w = (d4 - d3_) / ((d4 - d3_) + (d5 - d6));
        *cp = d3add(b, d3scale(d3sub(c, b), w));
        d3 v = d3sub(p, *cp);
        return d3dot(v, v);
    }

    double denom = 1.0 / (va + vb + vc);
    double vv = vb * denom, ww = vc * denom;
    *cp = d3add(d3add(a, d3scale(ab, vv)), d3scale(ac, ww));
    d3 v = d3sub(p, *cp);
    return d3dot(v, v);
}
