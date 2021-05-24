#ifndef CAMERA_H
#define CAMERA_H

#include "commons.h"

class camera; 
__global__ void free_camera_gpu(camera** d_camera);
__global__ void camera_gpu(camera** d_camera, vec3 w, vec3 u, vec3 v, point3 origin, vec3 horizontal, vec3 vertical, point3 lower_lef_corner, float lens_radius);

class camera {
public:

    __device__ __host__ camera() {

    }
    __device__ __host__ camera(
        point3 lookfrom,
        point3 lookat,
        vec3   vup,
        const float vfov,//vertical field of view in degrees
        const float aspect_ratio, 
        const float aperture,
        const float focus_dist
    ) {
        const float theta = vfov * PI / 180.0f;
        const float h = tanf(theta / 2.0f);
        const float viewport_height = 2.0f * h;
        const float viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperture / 2;
    }

    __device__ camera(vec3 w, vec3 u, vec3 v, point3 origin, vec3 horizontal, vec3 vertical, point3 lower_left_corner, float lens_radius)
     : w(w), u(u), v(v), origin(origin), horizontal(horizontal), vertical(vertical), lower_left_corner(lower_left_corner), lens_radius(lens_radius) {}

    __device__ ray get_ray(float s, float t, curandState* local_random_state) const {
        vec3 rd = lens_radius * random_in_unit_disk(local_random_state);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset
        );
    }

    __host__ void create_camera_on_gpu() {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(camera*)));
        camera_gpu << <1, 1 >> > (d_this, w, u, v, origin, horizontal, vertical, lower_left_corner, lens_radius);
    }

    __host__ __device__ ~camera() {
        #if !defined(__CUDA_ARCH__)
            if (d_this != NULL) {
                free_camera_gpu << <1, 1 >> > (d_this);
                checkCudaErrors(cudaFree(d_this));
            }
        #endif
    }

    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
    camera** d_this = NULL;
};

__global__ void camera_gpu(camera** d_camera, vec3 w, vec3 u, vec3 v, point3 origin, vec3 horizontal, vec3 vertical, point3 lower_lef_corner, float lens_radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_camera = new camera(w, u, v, origin, horizontal, vertical, lower_lef_corner, lens_radius);
    }
}

__global__ void free_camera_gpu(camera** d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete* d_camera;
    }
}

#endif 
