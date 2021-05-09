#ifndef COMMONS_H
#define COMMONS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <limits>

// Constants

#define INFINITY FLT_MAX
const float pi = 3.1415926535897932385;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

// Common Headers

#include "hittable.h"
#include "ray.h"
#include "vec3.h"

#endif