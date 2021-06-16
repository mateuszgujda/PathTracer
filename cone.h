#ifndef  CONE_H
#define CONE_H
#include "commons.h"

__global__ void cone_gpu(hittable** obj_ptr, material** mat_ptr, point3 center, float height, float radius);

//! Klasa zawieraj¹ca informacjê o obiektach sto¿kowych.
/**
 * Klasa zawieraj¹ca informacjê o obiektach sto¿kowych.
 */
class cone : public hittable {
public:
    /**
     * Konstruktor.
     * 
     * \return 
     */
    __device__ __host__ cone() {}
    /**
     * Konstruktor.
     * 
     * \param center Œrodek podstawy sto¿ka
     * \param height Wysokoœæ sto¿ka
     * \param radius Promieñ podstawy sto¿ka
     * \param m Matera³ u¿ywany w obiekcie
     * \return 
     */
    __device__ __host__ cone(point3 center, float height, float radius, material* m) : center(center), radius(radius), height(height) {
        this->material_ptr = m;
        this->tangens = (radius / height) * (radius / height);
    };

    //! @copydoc hittable::create_hittable_on_gpu()
    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(cone*)));
        cone_gpu << <1, 1 >> > (d_this, material_ptr->d_this, center, height, radius);
    }

    /**
     * Inicjalizacja z pliku.
     * 
     * \param is Stream do pliku
     * \param materials Lista materia³ów w scenie
     * \return Referencja do stworzonego obiektu
     */
    __host__ static cone* init_from_file(std::ifstream& is, std::vector<material*>& materials) {
        std::string line;
        std::streampos temp_pos;
        point3 cen;
        float r = 1.0f;
        float height = 10.0f;
        int material_index = 0;
        do {
            std::vector<std::string> maps;
            temp_pos = is.tellg();
            std::getline(is, line);
            maps = get_key_value(line);
            if (maps[0] == "center") {
                point3 p;
                p.load_from_string(maps[1]);
            }
            else if (maps[0] == "radius") {
                r = std::stof(maps[1]);
            }
            else if (maps[0] == "height") {
                height = std::stof(maps[1]);
            }
            else if (maps[0] == "material") {
                material_index = std::stoi(maps[1]);
            }
        } while (!isupper(line[0]));
        is.seekg(temp_pos);

        return new cone(cen, height, r, materials[material_index]);
    }

    //! @copydoc hittable::hit(r,t_min,t_max,rec)
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    //! @copydoc hittable::bounding_box(t0,t1,box)
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const override;

public:
    /**
     * Œrodek podstawy sto¿ka.
     */
    point3 center;
    /**
     * Wysokoœæ sto¿ka.
     */
    float height;
    /**
     * Promieñ sto¿ka.
     */
    float radius;
    /**
     * Tangens k¹tu sto¿ka.
     */
    float tangens;
};

__device__ bool cone::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

    vec3 oc = r.origin() - center;
    oc[1] = 0;
    vec3 ray_dir_xz = r.direction();
    ray_dir_xz[1] = 0;
    float D = height - r.origin().y() + center.y();
    float a = ray_dir_xz.length_squared() - tangens * r.direction().y() * r.direction().y();
    float half_b = dot(oc, ray_dir_xz) + tangens * r.direction().y() * D;
    float c = oc.length_squared() - tangens * D * D;

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
    if (rec.p.y() < center.y() || rec.p.y() > center.y() + height) {
        return false;
    }
    rec.material = material_ptr;
    vec3 outward_normal = (rec.p - center) / radius;
    vec3 pc = rec.p - center;
    pc[1] = 0;
    float normal_y = pc.length();
    outward_normal[1] = normal_y * radius/ height;
    rec.set_face_normal(r, outward_normal);

    return true;
}


__forceinline__ __device__ bool cone::bounding_box(float t0, float t1, aabb& box) const {

    box = aabb(center - vec3(radius, 0, radius),
        center + vec3(radius, height, radius));
    return true;
}



/**
 * Tworzenie \see cone na GPU.
 * 
 * \param obj_ptr WskaŸnik na obiekt
 * \param mat_ptr WskaŸnik na materia³
 * \param center Œrodek podstawy sto¿ka
 * \param height Wysokoœæ sto¿ka
 * \param radius Promieñ sto¿ka
 * \return 
 */
__global__ void cone_gpu(hittable** obj_ptr, material** mat_ptr, point3 center, float height, float radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *obj_ptr = new cone(center, height, radius, *mat_ptr);
    }
}

#endif