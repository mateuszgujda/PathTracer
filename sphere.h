#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"
#include "material.h"

__global__ void sphere_gpu(hittable** obj_ptr, material** mat_ptr, point3 center, float radius);

class sphere : public hittable {
public:
    __device__ sphere() {}
    __device__ __host__ sphere(point3 cen, float r, material* m) : center(cen), radius(r) {
        this->material_ptr = m;
    };

    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(sphere*)));
        sphere_gpu << <1, 1 >> > (d_this, material_ptr->d_this, center, radius);
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const override;

public:
    point3 center;
    float radius;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    rec.material = material_ptr;
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
   
    return true;
}


__forceinline__ __device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const {

    box = aabb(center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));
    return true;
}




__global__ void sphere_gpu(hittable** obj_ptr, material** mat_ptr, point3 center, float radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *obj_ptr = new sphere(center, radius, *mat_ptr);
    }
}

#endif