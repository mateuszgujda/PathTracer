#ifndef CAMERA_H
#define CAMERA_H

#include "commons.h"

class camera {
public:
    __device__ camera() {
        const float aspect_ratio = 16.0f / 9.0f;
        const float viewport_height = 2.0f;
        float viewport_width = aspect_ratio * viewport_height;
        const float focal_length = 1.0f;

        origin = point3(0.0f, 0.0f, 0.0f);
        horizontal = vec3(viewport_width, 0.0f, 0.0f);
        vertical = vec3(0.0f, viewport_height, 0.0f);
        lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - vec3(0, 0, focal_length);
    }
    __device__ ray get_ray(float u, float v) { return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin); }

    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};


#endif 
