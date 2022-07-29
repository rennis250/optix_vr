#pragma once

#include <cuda_runtime.h>

#include "vec.h"

__device__ float3 skyColor(float3 ray_direction) {
    float3 n = normalize(ray_direction);
    float t = (n.y + 1.0);
    float3 a = make_float3(1.0, 1.0, 1.0);
    float3 b = make_float3(0.5, 0.7, 1.0);
    return (1.0 - t) * a + t * b;
}

// from Mitsuba
float3 reinhardTonemap(float3 p, float key, float burn) {
    burn = min(1.0, max(1e-8, 1.0 - burn));

    float logAvgLuminance = logf(2.0);
    float maxLuminance = 50.0;
    float scale = key / logAvgLuminance;
    float Lwhite = maxLuminance * scale;

    /* Having the 'burn' parameter scale as 1/b^4 provides a nicely behaved knob */
    float invWp2 = 1.0 / (Lwhite * Lwhite * powf(burn, 4.0));

    /* Convert ITU-R Rec. BT.709 linear RGB to XYZ tristimulus values */
    float X = p.x * 0.412453 + p.y * 0.357580 + p.z * 0.180423;
    float Y = p.x * 0.212671 + p.y * 0.715160 + p.z * 0.072169;
    float Z = p.x * 0.019334 + p.y * 0.119193 + p.z * 0.950227;

    /* Convert to xyY */
    float normalization = 1.0 / (X + Y + Z);
    float x = X * normalization;
    float y = Y * normalization;
    float Lp = Y * scale;

    /* Apply the tonemapping transformation */
    Y = Lp * (1.0 + Lp * invWp2) / (1.0 + Lp);

    /* Convert back to XYZ */
    float ratio = Y / y;
    X = ratio * x;
    Z = ratio * (1.0 - x - y);

    /* Convert from XYZ tristimulus values to ITU-R Rec. BT.709 linear RGB */
    float3 outc;
    outc.x = 3.240479 * X + -1.537150 * Y + -0.498535 * Z;
    outc.y = -0.969256 * X + 1.875991 * Y + 0.041556 * Z;
    outc.z = 0.055648 * X + -0.204043 * Y + 1.057311 * Z;

    return outc;
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
