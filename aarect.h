#ifndef AARECT_H
#define AARECT_H

#include "commons.h"

#include "hittable.h"

__global__ void xy_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float y0, float y1, float k);
__global__ void xz_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float z0, float z1, float k);
__global__ void yz_rect_gpu(hittable** d_this, material** mat_ptr, float y0, float y1, float z0, float z1, float k);

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

class xz_rect : public hittable {
public:
    xz_rect() {}

    __host__ __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k,
        material* mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k) {
        material_ptr = mat;
    };

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(point3(x0, k- 0.0001f, z0), point3(x1, k+ 0.0001f, z1));
        return true;
    }

    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(xz_rect*)));
        xz_rect_gpu << <1, 1 >> > (d_this, material_ptr->d_this, x0, x1, z0, z1, k);
    }



public:
    float x0, x1, z0, z1, k;
};

class yz_rect : public hittable {
public:
    yz_rect() {}

    __host__ __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k,
        material* mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k) {
        material_ptr = mat;
    };

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(point3(k - 0.0001f, y0, z0), point3(k + 0.0001f, y1, z1));
        return true;
    }

    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(yz_rect*)));
        yz_rect_gpu << <1, 1 >> > (d_this, material_ptr->d_this, y0, y1, z0, z1, k);
    }



public:
    float y0, y1, z0, z1, k;
};

__global__ void xy_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float y0, float y1, float k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_this = new xy_rect(x0, x1, y0, y1, k, *mat_ptr);
    }
}

__global__ void xz_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float z0, float z1, float k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_this = new xz_rect(x0, x1, z0, z1, k, *mat_ptr);
    }
}

__global__ void yz_rect_gpu(hittable** d_this, material** mat_ptr, float y0, float y1, float z0, float z1, float k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_this = new yz_rect(y0, y1, z0, z1, k, *mat_ptr);
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

__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.material = material_ptr;
    rec.p = r.at(t);
    return true;
}

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max)
        return false;
    auto y = r.origin().y() + t * r.direction().y();
    auto z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.material = material_ptr;
    rec.p = r.at(t);
    return true;
}


#endif