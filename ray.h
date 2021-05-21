#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
    __device__ ray() {}
    __device__ ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction){}

    __device__ point3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }

    __device__ point3 at(float t) const {
        return orig + t * dir;
    }

public:
    point3 orig;
    vec3 dir;
};

using Ray = ray;

#endif