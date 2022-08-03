#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#define __both__ __host__ __device__

__forceinline__ __device__ float clamp(const float f, const float a, const float b)
{
    return fmaxf(a, fminf(f, b));
}

__forceinline__ __device__ float3 clamp(const float3& v, const float a, const float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __both__ float3 operator+(const float3& v1, const float3& v2)
{
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __both__ float3 operator-(const float3& v, const float f)
{
    return make_float3(v.x - f, v.y - f, v.z - f);
}

inline __both__ float3 operator-(const float3& v1, const float3& v2)
{
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline __both__ float3 operator*(const float3& v1, const float3& v2)
{
    return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

inline __both__ float3 operator/(const float3& v1, const float3& v2)
{
    return make_float3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

inline __both__ float3 operator*(const float s, const float3& v)
{
    return make_float3(s * v.x, s * v.y, s * v.z);
}

inline __both__ float3 operator/(const float3& v, const float s)
{
    return make_float3(v.x / s, v.y / s, v.z / s);
}

inline __both__ float3 operator*(const float3& v, const float s)
{
    return make_float3(v.x * s, v.y * s, v.z * s);
}

inline __both__ float3 cross(const float3 a, const float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
        -(a.x * b.z - a.z * b.x),
        a.x * b.y - a.y * b.x);
}

inline __both__ float dot(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __both__ float len(const float3 v)
{
    return sqrtf(dot(v, v));
}

inline __both__ float3 normalize(const float3 v)
{
    float l = 1.0 / len(v);
    return make_float3(v.x * l, v.y * l, v.z * l);
}

__both__ float3 reflect(const float3 v, const float3 n) {
    return v - 2.0 * dot(v, n) * n;
}

__device__ float3 Reflect(const float3 v, const float3 n) {
    return -1.0 * v + 2.0 * dot(v, n) * n;
}

inline float3 set_face_normal(const float3 ray_direction, const float3 outward_normal) {
    bool front_face = dot(ray_direction, outward_normal) < 0;
    return front_face ? outward_normal : -1.0 * outward_normal;
}

// ONB stuff

struct mat3 {
    float3 nn;
    float3 o1;
    float3 o2;
};

__device__ float3 ortho(float3 v) {
    return fabsf(v.x) > fabsf(v.z) ? make_float3(-v.y, v.x, 0.0) : make_float3(0.0, -v.z, v.y);
}

__device__ mat3 ONB(float3 n) {
    mat3 onb;
    onb.nn = normalize(n);
    onb.o1 = normalize(ortho(onb.nn));
    onb.o2 = normalize(cross(onb.o1, onb.nn));

    return onb;
}

__device__ mat3 transpose(mat3 onb) {
    mat3 onbt;

    onbt.nn = make_float3(onb.nn.x, onb.o1.x, onb.o2.x);

    onbt.o1 = make_float3(onb.nn.y, onb.o1.y, onb.o2.y);

    onbt.o2 = make_float3(onb.nn.z, onb.o1.z, onb.o2.z);

    return onbt;
}

__device__ float3 apply_onb(mat3 onb, float3 v) {
    return v.x * onb.o1 + v.y * onb.nn + v.z * onb.o2;
}

