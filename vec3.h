#ifndef VEC3_H
#define VEC3_H
#include "cuda_runtime.h"
#include <math.h>
#include <iostream>

//! Klasa reprezentująca wektor, kolor lub punkt w przestrzeni 3D.
/**
 * Klasa reprezentująca wektor, kolor lub punkt w przestrzeni 3D.
 */
class Vec3 {
	public:
        /**
         * Konstruktor wektora (0,0,0).
         */
        __host__ __device__ Vec3() {};

        /**
         * Konstruktor wektora.
         * 
         * \param e0 Pierwsza współrzędna 
         * \param e1 Druga współrzędna
         * \param e2 Trzecia współrzędna
         * \return 
         */
        __host__ __device__ Vec3(float e0, float e1, float e2) { 
            e[0] = e0;
            e[1] = e1;
            e[2] = e2;
        }
        /**
         * Zwrócenie pierwszego parametru wektora.
         *
         * \return Pierwszą liczbę w wektorze
         */
        __host__ __device__ inline float x() const { return e[0]; }
        /**
         * Zwrócenie drugiego parametru wektora.
         * 
         * \return Drugą liczbę w wektorze
         */
        __host__ __device__ inline float y() const { return e[1]; }
        /**
         * Zwrócenie trzeciego parametru wektora.
         * 
         * \return Trzecią liczbę w wektorze
         */
        __host__ __device__ inline float z() const { return e[2]; }
        /**
        * Zwrócenie pierwszego parametru wektora.
        *
        * \return Pierwszą liczbę w wektorze
        */
        __host__ __device__ inline float r() const { return e[0]; }
        /**
        * Zwrócenie drugiego parametru wektora.
        *
        * \return Drugą liczbę w wektorze
        */
        __host__ __device__ inline float g() const { return e[1]; }
        /**
        * Zwrócenie trzeciego parametru wektora.
        *
        * \return Trzecią liczbę w wektorze
        */
        __host__ __device__ inline float b() const { return e[2]; }

        /**
         * Przeładowany operator negacji wektora.
         * 
         * \return Zanegowany wektor
         */
        __host__ __device__ inline Vec3 operator-() const { 
            return Vec3(-e[0], -e[1], -e[2]); 
        }
        /**
         * Zwraca kopie współrzednej wektora zadaną parametrem.
         * 
         * \param i Numer współrzędnej wektora
         * \return Współrzędna wektora zadana liczbą i
         */
        __host__ __device__ inline float operator[](int i) const { return e[i]; }
        /**
        * Zwraca współrzedną wektora zadaną parametrem.
        *
        * \param i Numer współrzędnej wektora
        * \return Współrzędna wektora zadana liczbą i
        */
        __host__ __device__ inline float& operator[](int i) { return e[i]; }

        /**
         * Przeładowany operator +=.
         * 
         * \param v wektor który chcemy dodać
         * \return Wektor do którego dodany został wektor \param v
         */
        __host__ __device__ inline Vec3& operator+=(const Vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        };

        /**
         * Przeładowany operator *=.
         * 
         * \param t wartość saklująca
         * \return Wektor przeskalowany o \param t w każdym kierunku
         */
        __host__ __device__ inline Vec3& operator*=(const float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        };

        /**
         * Przeładowany operator *=.
         * 
         * \param v wektor, z którym chcemy pomnożyć 
         * \return Wektor z współrzędnymi przeskalowanymi przez współrzędne \param v
         */
        __host__ __device__ inline Vec3& Vec3::operator*=(const Vec3& v) {
            e[0] *= v.e[0];
            e[1] *= v.e[1];
            e[2] *= v.e[2];
            return *this;
        }

        /**
         * Przeładowany operator /=.
         * 
         * \param t wartość przez którą chcemy przeskalować wektor
         * \return Wektor pomniejszony \param t razy
         */
        __host__ __device__ inline Vec3& operator/=(const float t) {
            return *this *= 1 / t;
        };

        /**
         * Funkcja zwracająca długość wektora.
         * 
         * \return Długość wektora
         */
        __host__ __device__ inline float length() const {
            return sqrtf(length_squared());
        };

        /**
         * Funkcja zwracająca kwadrat długości wektora.
         * 
         * \return Kwadrat długości wektora
         */
        __host__ __device__ inline float length_squared() const {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        };

        /**
         * Funkcja sprawdzająca czy wektor jest bliski zera w każdym kierunku.
         * 
         * \return Prawde jeśli wektor jest bliski zera w każdym kierunku
         */
        __host__ __device__ bool near_zero() const {
            const float s = 1e-8f;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

        /**
         * Ładuje współrzędne wektora z ciągu znaków.
         * 
         * \param str ciąg znaków
         * \return 
         */
        __host__ void load_from_string(std::string str) {
            auto values = explode(str, ',');
            e[0] = stof(values[0]);
            e[1] = stof(values[1]);
            e[2] = stof(values[2]);
        }



	public:
        /**
         * Tablica współrzędnych wektora.
         */
		float e[3] = { 0.0f, 0.0f, 0.0f };



};

using point3 = Vec3;   // 3D point
using color = Vec3;    // RGB color
using vec3 = Vec3;     // vec3 alias

// vec3 Utility Functions

/**
 * Przeładowany operator do wipyswania wektora.
 * 
 * \param out stream do którego wypisujemy
 * \param v wektor, który jest wypisywany
 * \return stream do którego wypisujemy
 */
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

/**
 * Operator dodawania dwóch wektorów.
 * 
 * \param u Wektor do dodania
 * \param v Wektor do dodania
 * \return Nowy wektor o pozycjach zsumowanych z wektorów w parametrach
 */
__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

/**
 * Operator odejmowania dwóch wektorów.
 * 
 * \param u Wektor od, którego odejmujemy
 * \param v Wektor odejmowany
 * \return Nowy wektor o przeliczonych pozycjach
 */
__host__ __device__  inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

/**
 * Operator mnożenia wektorów.
 * 
 * \param u Wektor mnożony
 * \param v mnożnik
 * \return Nowy wektor, gdzie parametry to wymnożone wartości
 */
__host__ __device__  inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

/**
 * Operator mnożenia wektora przez skalar.
 * 
 * \param t liczba o którą skalowany jest wektor
 * \param v wektor który skalujemy
 * \return Nowy wektor przeskalowany o \param t
 */
__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

/**
 * Operator mnożenia wektora przez skalar.
 *
 * \param t liczba o którą skalowany jest wektor
 * \param v wektor który skalujemy
 * \return Nowy wektor przeskalowany o \param t
 */
__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

/**
 * Operator dzielenia wektora przez skalar.
 *
 * \param v wektor który skalujemy
 * \param t liczba o którą skalujemy jest wektor
 * \return Nowy wektor przeskalowany o \param t
 */
__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1 / t) * v;
}

/**
 * Operator iloczynu skalarnego.
 *
 * \param u 1 wektor
 * \param v 2 wektor
 * \return Wartośc iloczynu skalarnego
 */
__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

/**
 * Operator iloczynu wektorowego.
 *
 * \param u 1 wektor
 * \param v 2 wektor
 * \return Wartośc iloczynu wektorowego
 */
__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

/**
 * Wekor jednostkowy.
 * 
 * \param v Wektor z którego chcemy otrzymać wektor jednostkowy
 * \return wektor jednostkowy w kierunku tym co wektor \param v
 */
__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

/**
 * Odbicie wektora .
 * 
 * \param vector_on_surface wektor który uderza w powierzchnię
 * \param normal wektor normalny
 * \return Zwraca wektor odbity od powierzchni
 */
__host__ __device__ inline vec3 reflect(const vec3& vector_on_surface, const vec3& normal) {
    return vector_on_surface - 2 * dot(vector_on_surface, normal) * normal;
}

/**
 * Załamanie wektora.
 * 
 * \param uv wektor padający
 * \param n wektor normalny do powierzchni
 * \param etai_over_etat stosunek współczynnika załamania do powierzchni
 * \return Wektor po załamaniu
 */
__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif