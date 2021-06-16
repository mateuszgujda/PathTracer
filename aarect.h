#ifndef AARECT_H
#define AARECT_H

#include "commons.h"
#include <fstream>
#include "hittable.h"

__global__ void xy_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float y0, float y1, float k);
__global__ void xz_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float z0, float z1, float k);
__global__ void yz_rect_gpu(hittable** d_this, material** mat_ptr, float y0, float y1, float z0, float z1, float k);

/**
 * Klasa zawieraj�ca informacje o prostok�cie w p�aszczy�nie XY.
 */
class xy_rect : public hittable {
public:
    /**
     * Konstruktor.
     * 
     */
    xy_rect() {}

    /**
     * Konstruktor.
     * 
     * \param _x0 Pierwsza wsp�rz�dna w p�aszczy�nie X
     * \param _x1 Druga wsp�rz�dna w p�aszczy�nie X
     * \param _y0 Pierwsza wsp�rz�dna w p�aszczy�nie Y
     * \param _y1 Druga wsp�rz�dna wp�aszczy�nie Y
     * \param _k Szeroko�� prostok�ta
     * \param mat Materia� obiektu
     * \return 
     */
    __host__ __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k,
       material* mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k) {
        material_ptr = mat;
    };

    //! @copydoc hittable::hit(r,t_min,t_max,rec)
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    //! @copydoc hittable::bounding_box(time0,time1,output_box)
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(point3(x0, y0, k - 0.0001), point3(x1, y1, k + 0.0001));
        return true;
    }

    //! @copydoc hittable::create_hittable_on_gpu()
    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(xy_rect*)));
        xy_rect_gpu << <1, 1 >> > (d_this, material_ptr->d_this, x0, x1, y0, y1, k);
    }

    /**
     * Wczytywanie obiektu z pliku.
     * 
     * \param file Stream do pliku
     * \param materials Lista materia��w w scenie
     * \return Referencje do utworzonego obiektu
     */
    __host__ static xy_rect* load_from_file(std::ifstream& file, std::vector<material*>& materials) {
        std::string line;
        std::streampos temp_pos;
        float x0 = 0;
        float x1 = 1;
        float y0 = 0;
        float y1 = 1;
        float k = 0.2f;
        int material_index = 0;
        do {
            std::vector<std::string> maps;
            temp_pos = file.tellg();
            std::getline(file, line);
            maps = get_key_value(line);
            if (maps[0] == "x0") {
                x0 = std::stof(maps[1]);
            }
            else if (maps[0] == "x1") {
                x1 = std::stof(maps[1]);
            }
            else if (maps[0] == "y0") {
                y0 = std::stof(maps[1]);
            }
            else if (maps[0] == "y1") {
                y1 = std::stof(maps[1]);
            }
            else if (maps[0] == "k") {
                k = std::stof(maps[1]);
            }
            else if (maps[0] == "material") {
                material_index = std::stoi(maps[1]);
            }
        } while (!isupper(line[0]));
        file.seekg(temp_pos);

        return new xy_rect(x0, x1, y0, y1, k, materials[material_index]);
    }


public:
    /**
     * Wsp�rz�dne obektu.
     */
    float x0, x1, y0, y1, k;
};

/**
 * Klasa zawieraj�ca informacje o prostok�cie w p�aszczy�nie XZ.
 */
class xz_rect : public hittable {
public:
    /**
     * Konstrutor pusty.
     * 
     */
    xz_rect() {}

    /**
     * Konstruktor.
     * 
     * \param _x0 Pierwsza wsp�rz�dna w p�aszczy�nie X
     * \param _x1 Druga wsp�rz�dna w p�aszczy�nie X
     * \param _z0 Pierwsza wsp�rz�dna w p�aszczy�nie Z
     * \param _z1 Druga wsp�rz�dna wp�aszczy�nie Z
     * \param _k Szeroko�� prostok�ta
     * \param mat Materia� obiektu
     * \return 
     */
    __host__ __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k,
        material* mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k) {
        material_ptr = mat;
    };

    //! @copydoc hittable::hit(r,t_min,t_max,rec)
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    //! @copydoc hittable::bunding_box(time0, time1, output_box)
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(point3(x0, k- 0.0001f, z0), point3(x1, k+ 0.0001f, z1));
        return true;
    }

    //! @copydoc hittable::create_hittable_on_gpu()
    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(xz_rect*)));
        xz_rect_gpu << <1, 1 >> > (d_this, material_ptr->d_this, x0, x1, z0, z1, k);
    }

    /**
     * �adowanie obiektu z pliku.
     * 
     * \param file Stream do pliku
     * \param materials Lista u�ywanych materia��w
     * \return Referencja do utworzonego obiektu
     */
    __host__ static xz_rect* load_from_file(std::ifstream& file, std::vector<material*>& materials) {
        std::string line;
        std::streampos temp_pos;
        float x0 = 0;
        float x1 = 1;
        float z0 = 0;
        float z1 = 1;
        float k = 0.2f;
        int material_index = 0;
        do {
            std::vector<std::string> maps;
            temp_pos = file.tellg();
            std::getline(file, line);
            maps = get_key_value(line);
            if (maps[0] == "x0") {
                x0 = std::stof(maps[1]);
            }
            else if (maps[0] == "x1") {
                x1 = std::stof(maps[1]);
            }
            else if (maps[0] == "z0") {
                z0 = std::stof(maps[1]);
            }
            else if (maps[0] == "z1") {
                z1 = std::stof(maps[1]);
            }
            else if (maps[0] == "k") {
                k = std::stof(maps[1]);
            }
            else if (maps[0] == "material") {
                material_index = std::stoi(maps[1]);
            }
        } while (!isupper(line[0]));
        file.seekg(temp_pos);

        return new xz_rect(x0, x1, z0, z1, k, materials[material_index]);
    }



public:
    /**
     * Wsp�rz�dne obiektu.
     */
    float x0, x1, z0, z1, k;
};

/**
 * Klasa zawieraj�ca informacje o prostok�cie w p�aszczy�nie YZ.
 */
class yz_rect : public hittable {
public:
    yz_rect() {}

    __host__ __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k,
        material* mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k) {
        material_ptr = mat;
    };

    //! @copydoc hittable::hit(r,t_min,t_max,rec)
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    //! @copydoc hittable::bounding_box(time0,time1,output_box)
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = aabb(point3(k - 0.0001f, y0, z0), point3(k + 0.0001f, y1, z1));
        return true;
    }

    //! @copydoc hittable::create_hittable_on_gpu()
    __host__ virtual void create_hittable_on_gpu() override {
        checkCudaErrors(cudaMalloc(&d_this, sizeof(yz_rect*)));
        yz_rect_gpu << <1, 1 >> > (d_this, material_ptr->d_this, y0, y1, z0, z1, k);
    }

    /**
     * Wczytanie obiektu z pliku.
     * 
     * \param file Stream do pliku
     * \param materials Lista materia��w dost�pnych w scenie
     * \return Referencje do utworzonego obiektu
     */
    __host__ static yz_rect* load_from_file(std::ifstream& file, std::vector<material*>& materials) {
        std::string line;
        std::streampos temp_pos;
        float y0 = 0;
        float y1 = 1;
        float z0 = 0;
        float z1 = 1;
        float k = 0.2f;
        int material_index = 0;
        do {
            std::vector<std::string> maps;
            temp_pos = file.tellg();
            std::getline(file, line);
            maps = get_key_value(line);
            if (maps[0] == "y0") {
                y0 = std::stof(maps[1]);
            }
            else if (maps[0] == "y1") {
                y1 = std::stof(maps[1]);
            }
            else if (maps[0] == "z0") {
                z0 = std::stof(maps[1]);
            }
            else if (maps[0] == "z1") {
                z1 = std::stof(maps[1]);
            }
            else if (maps[0] == "k") {
                k = std::stof(maps[1]);
            }
            else if (maps[0] == "material") {
                material_index = std::stoi(maps[1]);
            }
        } while (!isupper(line[0]));
        file.seekg(temp_pos);

        return new yz_rect(y0, y1, z0, z1, k, materials[material_index]);
    }

public:
    /**
     * Wspo�rz�dne obiektu.
     */
    float y0, y1, z0, z1, k;
};

/**
 * Tworzenie obiektu \see xy_rect na GPU.
 * 
 * \param d_this wska�nik do obiektu
 * \param mat_ptr wska�nik na u�ywany materia�
 * \param x0 Pierwsza wsp�rz�dna w p�aszczy�nie X
 * \param x1 Druga wsp�rz�dna w p�aszczy�nie X
 * \param y0 Pirwsza wsp�rz�dna w p�aszczy�nie Y
 * \param y1 Druga wsp�rz�dna w p�aszczy�nie Y
 * \param k Szeroko�c prostok�ta
 * \return 
 */
__global__ void xy_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float y0, float y1, float k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_this = new xy_rect(x0, x1, y0, y1, k, *mat_ptr);
    }
}

/**
 * Tworzenie obiektu \see xz_rect na GPU.
 *
 * \param d_this wska�nik do obiektu
 * \param mat_ptr wska�nik na u�ywany materia�
 * \param x0 Pierwsza wsp�rz�dna w p�aszczy�nie X
 * \param x1 Druga wsp�rz�dna w p�aszczy�nie X
 * \param z0 Pirwsza wsp�rz�dna w p�aszczy�nie Z
 * \param z1 Druga wsp�rz�dna w p�aszczy�nie Z
 * \param k Szeroko�c prostok�ta
 * \return
 */
__global__ void xz_rect_gpu(hittable** d_this, material** mat_ptr, float x0, float x1, float z0, float z1, float k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_this = new xz_rect(x0, x1, z0, z1, k, *mat_ptr);
    }
}

/**
 * Tworzenie obiektu \see yz_rect na GPU.
 *
 * \param d_this wska�nik do obiektu
 * \param mat_ptr wska�nik na u�ywany materia�
 * \param y0 Pierwsza wsp�rz�dna w p�aszczy�nie Y
 * \param y1 Druga wsp�rz�dna w p�aszczy�nie Y
 * \param z0 Pirwsza wsp�rz�dna w p�aszczy�nie Z
 * \param z1 Druga wsp�rz�dna w p�aszczy�nie Z
 * \param k Szeroko�c prostok�ta
 * \return
 */
__global__ void yz_rect_gpu(hittable** d_this, material** mat_ptr, float y0, float y1, float z0, float z1, float k) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_this = new yz_rect(y0, y1, z0, z1, k, *mat_ptr);
    }
}

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.material = material_ptr;
    rec.p = r.at(t);
    return true;
}

__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.material = material_ptr;
    rec.p = r.at(t);
    return true;
}

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max)
        return false;
    auto y = r.origin().y() + t * r.direction().y();
    auto z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.material = material_ptr;
    rec.p = r.at(t);
    return true;
}


#endif