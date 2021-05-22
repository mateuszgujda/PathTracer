#ifndef AABB_H
#define AABB_H
#include "commons.h"
#include <thrust/swap.h>


__forceinline__ __device__ float ffmin(float a, float b) { return a < b ? a : b; }
__forceinline__ __device__ float ffmax(float a, float b) { return a > b ? a : b; }

class aabb {
public:
    __device__ aabb() {}
    __device__ aabb(const point3& a, const point3& b) { minimum = a; maximum = b; }

    __device__ point3 min() const { return minimum; }
    __device__ point3 max() const { return maximum; }

    bool aabb::hit(const ray& r, float t_min, float t_max) const;

    point3 minimum;
    point3 maximum;
};


__device__ inline bool aabb::hit(const ray& r, float t_min, float t_max) const {
    for (int a = 0; a < 3; a++) {
        auto invD = 1.0f / r.direction()[a];
        auto t0 = (min()[a] - r.origin()[a]) * invD;
        auto t1 = (max()[a] - r.origin()[a]) * invD;
        if (invD < 0.0f)
            thrust::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }
    return true;
}

__forceinline__ __device__ aabb surrounding_box(aabb box0, aabb box1) {
    vec3 small(ffmin(box0.min().x(), box1.min().x()),
        ffmin(box0.min().y(), box1.min().y()),
        ffmin(box0.min().z(), box1.min().z()));
    vec3 big(ffmax(box0.max().x(), box1.max().x()),
        ffmax(box0.max().y(), box1.max().y()),
        ffmax(box0.max().z(), box1.max().z()));
    return aabb(small, big);
}


#endif