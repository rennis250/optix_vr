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

__forceinline__ __device__ float3 toSRGB(const float3& c)
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}

__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)
{
    x = clamp(x, 0.0f, 1.0f);
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color(const float3& c)
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB(clamp(c, 0.0f, 1.0f));
    return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}

__forceinline__ __device__ uchar4 make_color(const float4& c)
{
    return make_color(make_float3(c.x, c.y, c.z));
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
