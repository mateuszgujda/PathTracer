#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "cuda_runtime.h"
#include "commons.h"
#include "aabb.h"

class material;
class hittable;
struct hit_record {
    point3 p;
    vec3 normal;
    material* material;
    float t;
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

__global__ void free_hittable_gpu(hittable** d_this);

class hittable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;

    __host__ virtual void create_hittable_on_gpu() = 0;
    __host__ __device__ ~hittable() {
        #if !defined(__CUDA_ARCH__)
            if (d_this != NULL) {
                checkCudaErrors(cudaFree(d_this));
            }
        #endif
    }
public:
    material* material_ptr;
    hittable** d_this = NULL;
};

__global__ void free_hittable_gpu(hittable** d_this) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete* d_this;
    }
}

#endif