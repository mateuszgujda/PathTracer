#ifndef AABB_H
#define AABB_H
#include "commons.h"
#include <thrust/swap.h>

/**
 * Funkcja min na GPU.
 * 
 * \param a pierwsza liczba
 * \param b druga liczba
 * \return Liczbe mniejsz¹ 
 */
__forceinline__ __device__ float ffmin(float a, float b) { return a < b ? a : b; }
/**
 * Funkcja max na GPU.
 * 
 * \param a pierwsza liczba
 * \param b druga liczba
 * \return Liczbe wiêksz¹
 */
__forceinline__ __device__ float ffmax(float a, float b) { return a > b ? a : b; }

/**
 * Klasa zawieraj¹ca szeœcian otaczaj¹cy dany obiekt.
 */
class aabb {
public:
    /**
     * Konstruktor.
     * 
     * \return 
     */
    __device__ aabb() {}
    /**
     * Konstruktor.
     * 
     * \param a róg szeœcianu
     * \param b róg szeœcianu po przek¹tnej od a
     * \return 
     */
    __device__ aabb(const point3& a, const point3& b) { minimum = a; maximum = b; }

    /**
     * Funkcja min.
     * 
     * \return Zwraca najbli¿szy punkt w szeœcianie
     */
    __device__ point3 min() const { return minimum;}
    /**
     * Funkcja max.
     * 
     * \return Zwraca najdalszy punkt w szeœcianie
     */
    __device__ point3 max() const { return maximum; }

    /**
     * Funkcja zajmuj¹ca siê trafieniem.
     * 
     * \param r promieñ uderzaj¹cy
     * \param t_min pocz¹tek przedzia³u czasowego
     * \param t_max koniec przedzia³u czasowego
     * \return  czy zosta³ trafiony
     */
    bool aabb::hit(const ray& r, float t_min, float t_max) const;

    /**
     * Punkt najbli¿szy.
     */
    point3 minimum;
    /**
     * Punkt najdalszy.
     */
    point3 maximum;
};


__device__ inline bool aabb::hit(const ray& r, float t_min, float t_max) const {
    for (int a = 0; a < 3; a++) {
        auto invD = 1.0f / r.direction()[a];
        auto t0 = (min()[a] - r.origin()[a]) * invD;
        auto t1 = (max()[a] - r.origin()[a]) * invD;
        if (invD < 0.0f)
            thrust::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }
    return true;
}

__forceinline__ __device__ aabb surrounding_box(aabb box0, aabb box1) {
    vec3 small(ffmin(box0.min().x(), box1.min().x()),
        ffmin(box0.min().y(), box1.min().y()),
        ffmin(box0.min().z(), box1.min().z()));
    vec3 big(ffmax(box0.max().x(), box1.max().x()),
        ffmax(box0.max().y(), box1.max().y()),
        ffmax(box0.max().z(), box1.max().z()));
    return aabb(small, big);
}


#endif