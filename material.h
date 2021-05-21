#ifndef MATERIAL_H
#define MATERIAL_H
#include "commons.h"
#include "hittable.h"
#include "cuda_functions.h"

class material;
__global__ void lambertian_gpu(material** mat_ptr, color albedo);
__global__ void metal_gpu(material** mat_ptr, color albedo, float fuzz);
__global__ void dielectric_gpu(material** mat_ptr, float ior);


class material {
    public :
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const = 0;

    __host__ virtual void create_material_on_gpu() = 0;

    __host__ __device__ ~material() {
        #if !defined(__CUDA_ARCH__)
        if (d_this != NULL) {
            checkCudaErrors(cudaFree(d_this));
        }
        #endif
    }
    public :
    material** d_this = NULL;
};

class lambertian : public material {
    public :
        __host__ __device__ lambertian(const color a) : albedo(a) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const override {
            vec3 scatter_direction = rec.normal + unit_vector(random_in_unit_sphere(local_rand_state));
            if (scatter_direction.near_zero()) {
                scatter_direction = rec.normal;
            }

            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        };

        __host__ virtual void create_material_on_gpu() override {
            checkCudaErrors(cudaMalloc(&d_this, sizeof(lambertian*)));
            lambertian_gpu << <1, 1 >> > (d_this, albedo);
        }

    public:
        color albedo;
};

class metal : public material {
public:
    __host__ __device__ metal(const color& a, float f) : albedo(a), fuzz(f > 1.0f ? 1.0f : f) {
    
    }

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

    __host__ virtual void create_material_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(metal*)));
        metal_gpu << <1, 1 >> > (d_this, albedo, fuzz);
    }

public:
    color albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __host__ __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const override {
        attenuation = color(1.0f, 1.0f, 1.0f);
        float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;
        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance_schlik_approximation(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

    __host__ virtual void create_material_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(dielectric*)));
        dielectric_gpu << <1, 1 >> > (d_this, ir);
    }

public:
    float ir; // Index of Refraction
};

__global__ void lambertian_gpu(material** mat_ptr, color albedo) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *mat_ptr = new lambertian(albedo);
    }
}


__global__ void metal_gpu(material** mat_ptr, color albedo, float fuzz) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *mat_ptr = new metal(albedo, fuzz);
    }
}

__global__ void dielectric_gpu(material** mat_ptr, float ior) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *mat_ptr = new dielectric(ior);
    }
}



#endif // !MATERIAL_H