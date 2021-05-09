#ifndef CAMERA_H
#define CAMERA_H

#include "commons.h"

class camera {
public:
    __device__ camera(
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

    __device__ ray get_ray(double s, double t, curandState* local_random_state) const {
        vec3 rd = lens_radius * random_in_unit_disk(local_random_state);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset
        );
    }

    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};


#endif 
