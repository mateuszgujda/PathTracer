﻿#ifndef VEC3_H
#define VEC3_H
#include "cuda_runtime.h"
#include <math.h>
#include <iostream>

class Vec3 {
	public:
        __host__ __device__ Vec3() {};
        __host__ __device__ Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; };
        __host__ __device__ inline float x() const { return e[0]; };
        __host__ __device__ inline float y() const { return e[1]; };
        __host__ __device__ inline float z() const { return e[2]; };
        __host__ __device__ inline float r() const { return e[0]; };
        __host__ __device__ inline float g() const { return e[1]; };
        __host__ __device__ inline float b() const { return e[2]; };

        __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); };
        __host__ __device__ inline float operator[](int i) const { return e[i]; };
        __host__ __device__ inline float& operator[](int i) { return e[i]; };

        __host__ __device__ inline Vec3& operator+=(const Vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        };

        __host__ __device__ inline Vec3& operator*=(const float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        };

        __host__ __device__ inline Vec3& operator/=(const float t) {
            return *this *= 1 / t;
        };

        __host__ __device__ inline float length() const {
            return sqrtf(length_squared());
        };

        __host__ __device__ inline float length_squared() const {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        };

	public:
		float e[3] = { 0.0f, 0.0f, 0.0f };



};

using point3 = Vec3;   // 3D point
using color = Vec3;    // RGB color
using vec3 = Vec3;     // vec3 alias

// vec3 Utility Functions

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__  inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__  inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}
#endif