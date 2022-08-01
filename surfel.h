#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <optix_device.h>

struct Surfel {
    float3 X;
    float3 wi;
    float t;
    float3 albedo;
    float3 position;
    float3 shadingNormal;
    int mat;
    bool hit;
    unsigned int seed;
};

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& p0, unsigned int& p1) {
    unsigned long long pre_ptr = reinterpret_cast<unsigned long long>(ptr);
    p0 = pre_ptr >> 32;
    p1 = pre_ptr & 0x00000000ffffffff;

    return;
}

static __forceinline__ __device__ void* unpackPointer() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();

    unsigned long long pre_ptr = static_cast<unsigned long long>(p0) << 32 | p1;
    void* ptr = reinterpret_cast<void*>(pre_ptr);

    return ptr;
}

static __forceinline__ __device__ Surfel* unpackSurfel() {
    return reinterpret_cast<Surfel*>(unpackPointer());
}
