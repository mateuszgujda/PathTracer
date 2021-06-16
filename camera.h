#ifndef CAMERA_H
#define CAMERA_H

#include "commons.h"

class camera; 
__global__ void free_camera_gpu(camera** d_camera);
__global__ void camera_gpu(camera** d_camera, vec3 w, vec3 u, vec3 v, point3 origin, vec3 horizontal, vec3 vertical, point3 lower_lef_corner, float lens_radius);

//! Klasa zawieraj¹ca informacje o kamerze w scenie.
/**
 * Klasa zawieraj¹ca informacje o kamerze w scenie.
 */
class camera {
public:
    
    /**
     * Konstruktor.
     * 
     * \return 
     */
    __device__ __host__ camera() {

    }

    /**
     * Konstruktor.
     * 
     * \param lookfrom Punkt z którego kamera patrzy
     * \param lookat Punkt na który kamera patrzy
     * \param vup Wektor wskazuj¹cy górê 
     * \param vfov K¹t widzenia kamery
     * \param aspect_ratio Proporcje obrazu kamery
     * \param aperture Œrednia soczewki
     * \param focus_dist Dystans przed rozmyciem
     * \return 
     */
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

    /**
     * Konstruktor.
     * 
     * \param w Wektor kamery
     * \param u Wektor kamery
     * \param v Wektor kamery
     * \param origin punkt, w którym znajduje siê kamera
     * \param horizontal wektor wzd³u¿ którego uk³ada siê szerokoœc obrazu
     * \param vertical wekto wzd³u¿ którego uk³ada siê wysokoœæ obrazu
     * \param lower_left_corner punkt, który wskazuje na lewy dolny róg obrazu w przestrzeni 3D
     * \param lens_radius œrednia soczewki
     * \return 
     */
    __device__ camera(vec3 w, vec3 u, vec3 v, point3 origin, vec3 horizontal, vec3 vertical, point3 lower_left_corner, float lens_radius)
     : w(w), u(u), v(v), origin(origin), horizontal(horizontal), vertical(vertical), lower_left_corner(lower_left_corner), lens_radius(lens_radius) {}

    /**
     * Funkcja licz¹ca promieñ dla wspó³rzêdnych s,t
     * 
     * \param s pierwsza wspó³rzêdna dla funkcji promienia
     * \param t druga wspo³rzêdna dla funkcji promienia
     * \param local_random_state Zmienna pozwalaj¹ca na tworzenie liczb pseudoloswych
     * \return Wygenerowany promieñ
     */
    __device__ ray get_ray(float s, float t, curandState* local_random_state) const {
        vec3 rd = lens_radius * random_in_unit_disk(local_random_state);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset
        );
    }

    /**
     * Tworzenie kamery na GPU.
     * 
     * \return 
     */
    __host__ void create_camera_on_gpu() {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(camera*)));
        camera_gpu << <1, 1 >> > (d_this, w, u, v, origin, horizontal, vertical, lower_left_corner, lens_radius);
    }

    /**
     * Destruktor.
     * 
     * \return 
     */
    __host__ __device__ ~camera() {
        #if !defined(__CUDA_ARCH__)
            if (d_this != NULL) {
                free_camera_gpu << <1, 1 >> > (d_this);
                checkCudaErrors(cudaFree(d_this));
            }
        #endif
    }

    /**
     * Punkt w którym znajduje siê kamera.
     */
    point3 origin;
    /**
     * Punkt lewego dolneo rogu obrazu w przestrzeni 3D.
     */
    point3 lower_left_corner;
    /**
     * wektor wzd³u¿ którego uk³ada siê szerokoœc obrazu.
     */
    vec3 horizontal;
    /**
     * wektor wzd³u¿ którego uk³ada siê wysokoœæ obrazu.
     */
    vec3 vertical;
    /**
     * Wektory wskazuj¹ce u³o¿enie kamery.
     */
    vec3 u, v, w;
    /**
     * Promieñ soczewki kamery.
     */
    float lens_radius;
    /**
     * WskaŸnik kamery na GPU.
     */
    camera** d_this = NULL;
};

/**
 * Tworzenie \see camera na GPU.
 * \param d_camera wskaŸnik na obiekt
 * \param w Wektor kamery
 * \param u Wektor kamery
 * \param v Wektor kamery
 * \param origin punkt, w którym znajduje siê kamera
 * \param horizontal wektor wzd³u¿ którego uk³ada siê szerokoœc obrazu
 * \param vertical wektor wzd³u¿ którego uk³ada siê wysokoœæ obrazu
 * \param lower_lef_corner punkt lewego dolnego rogu obrazu w przestrzeni 3D
 * \param lens_radius promieñ soczewki
 */
__global__ void camera_gpu(camera** d_camera, vec3 w, vec3 u, vec3 v, point3 origin, vec3 horizontal, vec3 vertical, point3 lower_lef_corner, float lens_radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_camera = new camera(w, u, v, origin, horizontal, vertical, lower_lef_corner, lens_radius);
    }
}

/**
 * Usuwanie \see caera z GPU.
 * \param d_camera wskaŸnik na obiekt kamery
 */
__global__ void free_camera_gpu(camera** d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete* d_camera;
    }
}

#endif 
