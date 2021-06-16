#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include "commons.h"
#include <curand_kernel.h>
#define PI 3.1415926535897932385f

/**
 * Zwraca losowe po³o¿enie na jednostkowej sferze.
 * 
 * \param local_rand_state zmienna umo¿liwiaj¹ca generowanie liczb pseudolosowych
 * \return 
 */
__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

/**
 * Zwraca pseudolosow¹ liczbê z zakresu na GPU.
 * 
 * \param local_rand_state zmienna umo¿liwiaj¹ca generowanie liczb pseudolosowych
 * \param min dolne ograniczenie losowanej liczby
 * \param max górne ograniczenie losowanej liczby
 * \return Wylosowana liczba
 */
__device__ int random_int(curandState* local_rand_state, int min, int max) {
    float myrandf = curand_uniform(local_rand_state);
    myrandf *= (max - min + 0.999999f);
    myrandf += min;
    return (int)truncf(myrandf);
}


__device__ vec3 random_in_hemisphere(const vec3& normal, curandState* local_rand_state) {
    vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

/**
 * Przyblio¿one odbicie przy u¿yciu wzoru Schilk'a.
 * 
 * \param kosinus k¹ta padania
 * \param refreaction_index wspó³czynnik za³amania
 * \return Wartoœæ k¹ta
 */
__device__ float reflectance_schlik_approximation(float cosine, float refreaction_index) {
    float r0 = (1.0f - refreaction_index) / (1.0f + refreaction_index);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
}

/**
 * Zwraca losowy punkt w dysku jednostkowym.
 * 
 * \param local_rand_state zmienna umo¿liwiaj¹ca generowanie liczb pseudolosowych
 * \return Punkt w dysku jednostkowym
 */
__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (p.length_squared() >= 1.0f);
    return p;
}

#endif // !CUDA_FUNCTIONS_H
