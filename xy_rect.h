#ifndef AARECT_H
#define AARECT_H

#include "commons.h"

#include "hittable.h"

__global__ void xy_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float y0, float y1, float k);

class xy_rect : public hittable {
public:
    xy_rect() {}

    __host__ __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k,
       material* mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k) {
        material_ptr = mat;
    };

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(point3(x0, y0, k - 0.0001), point3(x1, y1, k + 0.0001));
        return true;
    }

    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(xy_rect*)));
        xy_rect_gpu << <1, 1 >> > (d_this, material_ptr->d_this, x0, x1, y0, y1, k);
    }



public:
    float x0, x1, y0, y1, k;
};

__global__ void xy_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float y0, float y1, float k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_this = new xy_rect(x0, x1, y0, y1, k, *mat_ptr);
    }
}

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.material = material_ptr;
    rec.p = r.at(t);
    return true;
}

#endif