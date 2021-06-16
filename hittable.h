#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "cuda_runtime.h"
#include "commons.h"
#include "aabb.h"

class material;
class hittable;
/**
 * Struktura zawieraj¹ca dane o uderzeniu promienia.
 */
struct hit_record {
    point3 p;
    vec3 normal;
    material* material;
    float t;
    bool front_face;

    /**
     * Funkcja ustawiaj¹ca kierunek wektora normalnego.
     * 
     * \param r promieñ uderzaj¹cy
     * \param outward_normal wektor normalny "na zwen¹trz"
     * \return 
     */
    __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

__global__ void free_hittable_gpu(hittable** d_this);

/**
 * Klasa opisuj¹ca obiekty które mog¹ byæ po³o¿one w scenie.
 */
class hittable {
public:
    /**
     * Funkcja odpowiadaj¹ca za sprawdzenie czy promieñ uderza w obiekt.
     * 
     * \param r Promieñ uderzaj¹cy
     * \param t_min Pocz¹tek przedzia³u czasowy dla poruszaj¹cych obiektów
     * \param t_max Koniec  przedzia³u czasowego dla poruszaj¹cego obiektu
     * \param rec Informacje o trafieniu
     * \return Prawde je¿eli promien \param r uderzy³ w obiekt
     */
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    /**
     * Funkcja odpowiadaj¹ca za sprawdzenie szeœciennego otoczenia obiektu.
     * 
     * \param time0 Pocz¹tek przedzia³u czasowy dla poruszaj¹cych obiektów
     * \param time1 Koniec  przedzia³u czasowego dla poruszaj¹cego obiektu
     * \param output_box szeœcian, który zosta³ trafiony
     * \return Prawde je¿eli otoczenie zosta³o trafione
     */
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;

    /**
     * Funkcja odpowiedzialna za stworzenie obiektu na GPU.
     * 
     * \return 
     */
    __host__ virtual void create_hittable_on_gpu() = 0;
    /**
     * Destruktor.
     * 
     * \return 
     */
    __host__ __device__ ~hittable() {
        #if !defined(__CUDA_ARCH__)
            if (d_this != NULL) {
                checkCudaErrors(cudaFree(d_this));
            }
        #endif
    }
public:
    /**
     * WskaŸnik na u¿ywany przez obiekt materia³.
     */
    material* material_ptr;
    /**
     * WskaŸnik na obiekt na GPU.
     */
    hittable** d_this = NULL;
};

/**
 * Kernel usuwaj¹cy obiekt z GPU.
 * 
 * \param d_this WskaŸnik na obiekt
 * \return 
 */
__global__ void free_hittable_gpu(hittable** d_this) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete* d_this;
    }
}

#endif