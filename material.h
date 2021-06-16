#ifndef MATERIAL_H
#define MATERIAL_H
#include "commons.h"
#include "hittable.h"
#include "cuda_functions.h"
#include <iostream>
#include <fstream>
#include <string>
#include "commons.h"


class material;

__global__ void lambertian_gpu(material** mat_ptr, color albedo);
__global__ void metal_gpu(material** mat_ptr, color albedo, float fuzz);
__global__ void dielectric_gpu(material** mat_ptr, float ior);
__global__ void diffuse_light_gpu(material** mat_ptr, color albedo);
__global__ void delete_material_gpu(material** mat_ptr);
/**
* Podstawowa klasa materia�u, kt�r� powinny rozszerza� inne materia�y.
*/
class material {
    public :
    /**
     * Funkcja odpowiadaj�ca za obliczenia odbicia promienia od materia�u.
     * 
     * \param r_in Promie�, kt�ry uderza w materia�
     * \param rec Informacja o trafionym obiekcie
     * \param attenuation Kolor odbicia
     * \param scattered Odbity promie�
     * \param local_rand_state Zmienna odpowiedzialna za generowanie liczb pseudolosowych
     * \return Czy nast�pi�o odbicie
     */
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const = 0;

    /**
     * Funkcja dla obiekt�w zwracaj�cych �wiat�o. Domy�lnie zwraca czarny kolor
     * 
     * \param p Punkt trafienia
     * \return Kolor emitowanego �wiat�a
     */
    __device__ virtual color emitted(const point3& p) const {
        return color(0.0f, 0.0f, 0.0f);
    }

    /**
     * Funkcja odpowiedzialna za tworzenie materia�u na GPU.
     * 
     * \return 
     */
    __host__ virtual void create_material_on_gpu() = 0;

    /**
     * Destruktor.
     * 
     * \return 
     */
    __host__ __device__ ~material() {
        #if !defined(__CUDA_ARCH__)
        if (d_this != NULL) {
            delete_material_gpu << <1, 1 >> > (d_this);
            checkCudaErrors(cudaFree(d_this));
        }
        #endif
    }
    public :
    /**
     * Wska�nik na materia� na GPU.
     */
    material** d_this = NULL;
};

/**
 * Klasa materia��w o sta�ym kolorze.
 */
class lambertian : public material {
    public :
        /**
         * Konstruktor.
         * 
         * \param a Kolor materia�u
         */
        __host__ __device__ lambertian(const color a) : albedo(a) {}

        //! @copydoc material::scatter(r_in,rec,attenuation,scattered,local_rand_state)
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const override {
            vec3 scatter_direction = rec.normal + unit_vector(random_in_unit_sphere(local_rand_state));
            if (scatter_direction.near_zero()) {
                scatter_direction = rec.normal;
            }

            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        };

        /**
         * Wczytywanie materia�u z pliku.
         * 
         * \param file Stream do pliku
         * \return Referencja do utworzongo materia�u
         */
        __host__ static lambertian* load_from_file(std::ifstream& file) {
            std::string line;
            std::streampos temp_pos;
            color albedo;
            do {
                std::vector<std::string> maps;
                temp_pos = file.tellg();
                std::getline(file, line);
                maps = get_key_value(line);
                if (maps[0] == "albedo") {
                    albedo.load_from_string(maps[1]);
                }
            } while (!isupper(line[0]));
            file.seekg(temp_pos);
            return new lambertian(albedo);
        }

        //! @copydoc material::create_material_on_gpu()
        __host__ virtual void create_material_on_gpu() override {
            checkCudaErrors(cudaMalloc(&d_this, sizeof(lambertian*)));
            lambertian_gpu << <1, 1 >> > (d_this, albedo);
        }

    public:
        /**
         * Kolor materia�u.
         */
        color albedo;
};

/**
 * Klasa materia��w odbijaj�cych.
 */
class metal : public material {
public:
    /**
     * Konstruktor klasy.
     * 
     * \param a Kolor materia�u
     * \param f Zaburzenie odbicia
     * \return 
     */
    __host__ __device__ metal(const color& a, float f) : albedo(a), fuzz(f > 1.0f ? 1.0f : f) {
    
    }

    //! @copydoc material::scatter(r_in,rec,attenuation,scattered,local_rand_state)
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

    /**
     * Tworzenie materia�u z pliku.
     * 
     * \param is Stream do pliku
     * \return Referencja do utworzonego obiektu
     */
    __host__ static metal* load_from_file(std::ifstream& is) {
        std::string line;
        std::streampos temp_pos;
        color albedo;
        float fuzz = 0.5f;
        do {
            std::vector<std::string> maps;
            temp_pos = is.tellg();
            std::getline(is, line);
            maps = get_key_value(line);
            if (maps[0] == "albedo") {
                albedo.load_from_string(maps[1]);
            }
            else if (maps[0] == "fuzz") {
                fuzz = std::stof(maps[1]);
            }
        } while (!isupper(line[0]));
        is.seekg(temp_pos);
        return new metal(albedo, fuzz);
    }

    //! @copydoc material::create_material_on_gpu()
    __host__ virtual void create_material_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(metal*)));
        metal_gpu << <1, 1 >> > (d_this, albedo, fuzz);
    }

public:
    /**
     * Kolor materia�u.
     */
    color albedo;
    /**
     * Rozmycie odbicia.
     */
    float fuzz;
};


/**
 * Klasa materi��w prze�roczystych.
 */
class dielectric : public material {
public:
    /**
     * Konstruktor.
     * 
     * \param index_of_refraction Wsp�czynnik za�amania powierchni dla materia�u
     * \return 
     */
    __host__ __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    //! @copydoc material::scatter(r_in,rec,attenuation,scattered,local_rand_state)
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const override {
        attenuation = color(1.0f, 1.0f, 1.0f);
        float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;
        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance_schlik_approximation(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

    /**
     * Wczytanie z pliku.
     * 
     * \param is Stream do pliku
     * \return Referencja do utwozonego obiektu
     */
    __host__ static dielectric* load_from_file(std::ifstream& is) {
        std::string line;
        std::streampos temp_pos;
        float ior = 1.0f;
        do {
            std::vector<std::string> maps;
            temp_pos = is.tellg();
            std::getline(is, line);
            maps = get_key_value(line);
            if (maps[0] == "ior") {
                ior = std::stof(maps[1]);
            }
        } while (!isupper(line[0]));
        is.seekg(temp_pos);
        return new dielectric(ior);
    }

    //! @copydoc material::create_material_on_gpu()
    __host__ virtual void create_material_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(dielectric*)));
        dielectric_gpu << <1, 1 >> > (d_this, ir);
    }

public:
    /**
     * Wsp�czynnik za�amania promienia.
     */
    float ir; 
};

/**
 * Klasa materia��w emituj�cych �wiat�o.
 */
class diffuse_light : public material {
public:
    /**
     * Konstruktor.
     * 
     * \param c Kolor emitowanego �wiat�a
     * \return 
     */
    __host__ __device__ diffuse_light(color c) : emit(c) {}

    //! @copydoc material::scatter(r_in,rec,attenuation,scattered,local_rand_state)
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const override {
        return false;
    }

    //! @copydoc material::emitted(p)
    __device__ virtual color emitted(const point3& p) const override {
        return emit;
    }

    //! @copydoc material::create_material_on_gpu()
    __host__ virtual void create_material_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(dielectric*)));
        diffuse_light_gpu << <1, 1 >> > (d_this, emit);
    }

    /**
     * Tworzenie materia�u z pliku.
     * 
     * \param is Stream do pliku
     * \return Referencja do utworzonego materia�u
     */
    __host__ static diffuse_light* load_from_file(std::ifstream& is) {
        std::string line;
        std::streampos temp_pos;
        color emit;
        do {
            std::vector<std::string> maps;
            temp_pos = is.tellg();
            std::getline(is, line);
            maps = get_key_value(line);
            if (maps[0] == "emit") {
                emit.load_from_string(maps[1]);
            }
        } while (!isupper(line[0]));
        is.seekg(temp_pos);
        return new diffuse_light(emit);
    }

public:
    /**
     * Kolor emitowanego �wiat�a.
     */
    color emit;
};

/**
 * Tworzenie materia�u \see lambertian na GPU.
 * 
 * \param mat_ptr Wska�nik do materia�u
 * \param albedo Kolor materia�u
 * \return 
 */
__global__ void lambertian_gpu(material** mat_ptr, color albedo) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *mat_ptr = new lambertian(albedo);
    }
}

/**
 * Tworzenie materia�u \see metal na GPU.
 * 
 * \param mat_ptr Wska�nik do materia�u
 * \param albedo Kolor materia�u
 * \param fuzz Rozmycie odbicia
 * \return 
 */
__global__ void metal_gpu(material** mat_ptr, color albedo, float fuzz) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *mat_ptr = new metal(albedo, fuzz);
    }
}

/**
 * Tworzenie materia�u \see dielectric na GPU.
 * 
 * \param mat_ptr Wska�nik do materia�u
 * \param ior Wsp�czynnik za�amania
 * \return 
 */
__global__ void dielectric_gpu(material** mat_ptr, float ior) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *mat_ptr = new dielectric(ior);
    }
}

/**
 * Tworzenie materia�u \see diffuse_light na GPU.
 * 
 * \param mat_ptr Wska�nik do materia�u
 * \param albedo Kolor �wiat�a
 * \return 
 */
__global__ void diffuse_light_gpu(material** mat_ptr, color albedo) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *mat_ptr = new diffuse_light(albedo);
    }
}

/**
 * Usuwanie materia�u z GPU.
 * 
 * \param mat_ptr Wska�nik do materia�u
 * \return 
 */
__global__ void delete_material_gpu(material** mat_ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete* mat_ptr;
    }
}


#endif // !MATERIAL_H