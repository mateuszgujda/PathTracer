#ifndef MATERIAL_H
#define MATERIAL_H
#include "commons.h"
#include "hittable.h"
#include "cuda_functions.h"

class material {
    public :
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const = 0;
};

class lambertian : public material {
    public :
        __device__ lambertian(const color &a) : albedo(a) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const override {
            vec3 scatter_direction = rec.normal + unit_vector(random_in_unit_sphere(local_rand_state));
            if (scatter_direction.near_zero()) {
                scatter_direction = rec.normal;
            }

            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        };

    public:
        color albedo;
};

class metal : public material {
public:
    __device__ metal(const color& a, float f) : albedo(a), fuzz(f > 1.0f ? 1.0f : f) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

public:
    color albedo;
    float fuzz;
};

#endif // !MATERIAL_H
