#ifndef COMMONS_H
#define COMMONS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <string>
#include <iostream>
// Constants

#define INFINITY FLT_MAX
const float pi = 3.1415926535897932385;



// Utility Functions

/**
 * Zamiana stopni na radiany.
 * 
 * \param degrees liczba stopni
 * \return Liczbe stopni w radianach
 */
inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

/**
 * Funkcja pseudoloswa dla liczb.
 * 
 * \return Zwraca pseudolosow¹ liczbê z przedzia³u [0,1)
 */
inline float random_float() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0f);
}

/**
 * Funkcja losuj¹ca liczbê z przedzia³u.
 * 
 * \param min Dolne ograniczenie losowania
 * \param max Górne ograniczenie losowania
 * \return Zwraca pseudolosow¹ liczbê z przedzia³u [\param min, \param max)
 */
inline float random_float (float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

/**
 * Funkcja obcinaj¹ca liczbê do przedzia³u.
 * 
 * \param x Liczba do obciêcia
 * \param min wartoœæ minimalna
 * \param max wartoœæ maksyalna
 * \return Liczba po obciêciu
 */
inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}


/**
 * Funkcja do sprawdzenia b³êdów i ich wypisywania na GPU.
 */
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Common Headers

#include "str_fun.h"
#include "hittable.h"
#include "ray.h"
#include "vec3.h"




#endif