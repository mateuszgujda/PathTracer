#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"
#include "material.h"

__global__ void sphere_gpu(hittable** obj_ptr, material** mat_ptr, point3 center, float radius);

//! Klasa obiektu typu kula
class sphere : public hittable {
public:

    /**
     * Konstruktor pusty.
     * 
     */
    __device__ __host__ sphere() {}
    /**
     * Konstruktor klasy.
     * 
     * \param cen Po³o¿enie centrum kuli
     * \param r Wielkoœc promienia
     * \param m Materia³ kuli
     */
    __device__ __host__ sphere(point3 cen, float r, material* m) : center(cen), radius(r) {
        this->material_ptr = m;
    };

    //! @copydoc hittable::create_hittable_on_gpu()
    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(sphere*)));
        sphere_gpu << <1, 1 >> > (d_this, material_ptr->d_this, center, radius);
    }

    //! @copydoc hittable::hit(r,t_min,t_max,rec)
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    //! @copydoc hittable::bounding_box(t0,t1,box)
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const override;

    /**
     * Inicjalizacja obiektu z pliku.
     * 
     * \param is Stream do pliku
     * \param materials Lista dostêpnych materia³ów w scenie
     * \return Referencja do utworzonego obiektu
     */
    __host__ static sphere* init_from_file(std::ifstream& is, std::vector<material*>& materials) {
        std::string line;
        std::streampos temp_pos;
        point3 cen;
        float r = 1.0f;
        int material_index = 0;
        do {
            std::vector<std::string> maps;
            temp_pos = is.tellg();
            std::getline(is, line);
            maps = get_key_value(line);
            if (maps[0] == "center") {
                cen.load_from_string(maps[1]);
            }
            else if (maps[0] == "radius") {
                r = std::stof(maps[1]);
            }
            else if (maps[0] == "material") {
                material_index = std::stoi(maps[1]);
            }
        } while (!isupper(line[0]));
        is.seekg(temp_pos);

        return new sphere(cen, r, materials[material_index]);
    }

public:
    /**
     * Parametr opisuj¹cy œrodek kuli.
     */
    point3 center;
    /**
     * Promieñ kuli.
     */
    float radius;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    rec.material = material_ptr;
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
   
    return true;
}

__forceinline__ __device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const {

    box = aabb(center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));
    return true;
}



/**
 * Tworzenie obiektu \see sphere na GPU.
 * 
 * \param obj_ptr wskaŸnik na obiekt
 * \param mat_ptr wskaŸnik na materia³
 * \param center œrodek kuli w przestrzeni 3D
 * \param radius promieñ kuli
 * \return 
 */
__global__ void sphere_gpu(hittable** obj_ptr, material** mat_ptr, point3 center, float radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *obj_ptr = new sphere(center, radius, *mat_ptr);
    }
}

#endif