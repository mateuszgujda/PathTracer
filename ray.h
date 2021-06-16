#ifndef RAY_H
#define RAY_H

#include "vec3.h"

//! Klasa promienia
/**
 * Klasa zawieraj¹ca informacje o promieniu.
 */
class ray {
public:
    /**
     * Konstruktor.
     * 
     * \return 
     */
    __device__ ray() {}
    /**
     * Konstruktor.
     * 
     * \param origin Punkt wyjcia promienia
     * \param direction Kierunek promienia
     * \return 
     */
    __device__ ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction){}

    /**
     * Funkcja zwracaj¹ca punkt wyjœcia promienia.
     * 
     * \return Punkt wyjœcia
     */
    __device__ point3 origin() const { return orig; }
    /**
     * Funkcja zwracaj¹ca kierune promienia.
     * 
     * \return 
     */
    __device__ vec3 direction() const { return dir; }

    /**
     * Funkcja zwracaj¹ca promieñ w momencie t.
     * 
     * \param t moment czasowy
     * \return punkt dla \param t
     */
    __device__ point3 at(float t) const {
        return orig + t * dir;
    }

public:
    /**
     * Punkt wyjœcia promienia.
     */
    point3 orig;
    /**
     * Kireunek promienia.
     */
    vec3 dir;
};

using Ray = ray;

#endif