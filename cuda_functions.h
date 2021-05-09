#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "commons.h"
#include <curand_kernel.h>

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ vec3 random_in_hemisphere(const vec3& normal, curandState* local_rand_state) {
    vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}


#endif // !CUDA_FUNCTIONS_H
