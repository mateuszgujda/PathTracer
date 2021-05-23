#ifndef  CYLINDER_H
#define CYLINDER_H
#include "commons.h"

__global__ void cylinder_gpu(hittable** obj_ptr, material** mat_ptr, point3 center, float height, float radius);

class cylinder2 : public hittable {
public:
    __device__ cylinder2() {}
    __device__ __host__ cylinder2(point3 center, float height, float radius, material* m) : center(center), radius(radius), height(height) {
        this->material_ptr = m;
    };

    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(cylinder2*)));
        cylinder_gpu << <1, 1 >> > (d_this, material_ptr->d_this, center,height, radius);
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const override;

public:
    point3 center;
    float height;
    float radius;
};

__device__ bool cylinder2::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

    float half_height = height / 2;
    vec3 oc = r.origin() - center;
    oc[1] = 0;
    vec3 ray_dir_xz = r.direction();
    ray_dir_xz[1] = 0;
    float a = ray_dir_xz.length_squared();
    float half_b = dot(oc, ray_dir_xz);
    float c = oc.length_squared() - (radius * radius);

    float discriminant = half_b * half_b -  a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (- half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (- half_b + sqrtd) /  a;
        if (root < t_min || t_max < root)
            return false;
    }


    rec.t = root;
    rec.p = r.at(rec.t);
    if (rec.p.y() < center.y() || rec.p.y() > center.y() + height) {
        return false;
    }
    rec.material = material_ptr;
    vec3 outward_normal = (rec.p - center) / radius;
    outward_normal[1] = rec.p.y();
    rec.set_face_normal(r, outward_normal);

    return true;
}


__forceinline__ __device__ bool cylinder2::bounding_box(float t0, float t1, aabb& box) const {

    box = aabb(center - vec3(radius, 0, radius),
        center + vec3(radius, height, radius));
    return true;
}




__global__ void cylinder_gpu(hittable** obj_ptr, material** mat_ptr, point3 center, float height, float radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *obj_ptr = new cylinder2(center, height, radius, *mat_ptr);
    }
}

#endif