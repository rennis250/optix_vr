#pragma once

#define EPS 0.0001

#define Pi 3.14159265358979323846
#define InvPi 0.31830988618379067154
#define Inv2Pi 0.15915494309189533577
#define Inv4Pi 0.07957747154594766788
#define PiOver2 1.57079632679489661923
#define PiOver4 0.78539816339744830961
#define Sqrt2 1.41421356237309504880

#include <cuda_runtime.h>

#include "vec.h"
#include "surfel.h"
#include "rand.h"

float2 ConcentricSampleDisk(unsigned int &seed) {
    float2 u = make_float2(rnd(seed), rnd(seed));
    float2 uOffset = make_float2(2.0 * u.x - 1.0, 2.0 * u.y - 1.0);
    if (uOffset.x == 0.0 && uOffset.y == 0.0) {
        return make_float2(0.0, 0.0);
    }

    float theta, r;
    if (fabsf(uOffset.x) > fabsf(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }

    return make_float2(r * cosf(theta), r * sinf(theta));
}

float3 cosWeightHemi(unsigned int &seed) {
    float2 d = ConcentricSampleDisk(seed);
    float z = sqrtf(max(0.0, 1.0 - d.x * d.x - d.y * d.y));
    return make_float3(d.x, z, d.y);
}

float FrConductor(float cosThetaI, float etai, float etat, float k) {
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);
    float eta = etat / etai;
    float etak = k / etai;

    float cosThetaI2 = cosThetaI * cosThetaI;
    float sinThetaI2 = 1.0 - cosThetaI2;
    float eta2 = eta * eta;
    float etak2 = etak * etak;

    float t0 = eta2 - etak2 - sinThetaI2;
    float a2plusb2 = sqrtf(t0 * t0 + 4.0 * eta2 * etak2);
    float t1 = a2plusb2 + cosThetaI2;
    float a = sqrtf(0.5 * (a2plusb2 + t0));
    float t2 = 2.0 * cosThetaI * a;
    float Rs = (t1 - t2) / (t1 + t2);

    float t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    float t4 = t2 * sinThetaI2;
    float Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (Rp + Rs);
}

enum Material {
    METAL = 0,
    LAMB
};

__device__ float3 finiteScatteringDensity(Surfel &surfelX, float3 woW, float &eta_for_RR, float &pdf) {
    int mat = surfelX.mat;
    float3 X = surfelX.position;
    float3 n = surfelX.shadingNormal;

    mat3 ltow = ONB(n);
    mat3 wtol = transpose(ltow);

    if (mat == LAMB) {
        surfelX.wi = apply_onb(ltow, cosWeightHemi(surfelX.seed));
        if (dot(surfelX.wi, n) > 0.0 && dot(woW, n) > 0.0) {
            surfelX.position = surfelX.position + n * EPS;
            pdf = fabs(dot(surfelX.wi, n)) * InvPi;
            return surfelX.albedo * InvPi;
        } else {
            pdf = 0.0;
            return make_float3(0.0, 0.0, 0.0);
        }
    } else if (mat == METAL) {
        surfelX.position = surfelX.position + n * EPS;
        surfelX.wi = Reflect(normalize(woW), n);
        pdf = 1.0;
        return surfelX.albedo * FrConductor(fabs(dot(surfelX.wi, n)), 1.0, 1.4, 1.6) / fabs(dot(surfelX.wi, n));
    }
}