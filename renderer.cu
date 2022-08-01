#include <optix.h>
#include <cuda_runtime.h>
#include <optix_device.h>

#include "launch_params.h"
#include "vec.h"
#include "surfel.h"
#include "color.h"
#include "materials.h"

// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

extern "C" {
    __constant__ Params params;
}

__device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction, unsigned int seed)
{
    float3 U = params.cam_u;
    float3 V = params.cam_v;
    float3 W = params.cam_w;

    // The center of each pixel is at fraction (0.5,0.5)
    const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

    // Normalizing coordinates to [-1.0, 1.0]
    float3 d = 2.0f * make_float3(
        (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(dim.x),
        (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(dim.y),
        0.0f
    ) - 1.0f;

    origin = params.cam_eye;
    direction = normalize(d.x * U + d.y * V + W);

    return;
}

__device__ float3 L_i(Surfel &surfelY, float3 ray_origin, float3 ray_dir, unsigned int p0, unsigned int p1) {
    float3 L = make_float3(0.0, 0.0, 0.0);
    float3 beta = make_float3(1.0, 1.0, 1.0);
    float pdf = 0.0;
    float3 fr, wo;
    float eta_for_RR = 0.0;
    float etaScale = 0.0;

    float3 X = ray_origin;
    surfelY.wi = ray_dir;

    int nbounces = 10;
    for (int i = 0; i < nbounces; i++) {
        optixTrace(
            params.handle,
            X,
            surfelY.wi,
            0.0f,   // Min intersection distance
            1e16f,  // Max intersection distance
            0.0f,   // ray-time -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            SURFACE_RAY_TYPE,      // SBT offset -- See SBT discussion
            RAY_TYPE_COUNT,      // SBT stride -- See SBT discussion 
            SURFACE_RAY_TYPE,      // missSBTIndex -- See SBT discussion
            p0, p1); // These 32b values are the ray payload

        if (surfelY.hit) {
            wo = -1.0 * surfelY.wi;

            // L = L + beta * emittedRadiance(surfelY, wo);

            fr = finiteScatteringDensity(surfelY, wo, eta_for_RR, pdf);

            X = surfelY.position;

            if (pdf == 0.0) {
                break;
            }

            beta = beta * (fr * fabsf(dot(surfelY.wi, surfelY.shadingNormal))) / pdf;

            // Update the term that tracks radiance scaling for refraction
            // depending on whether the ray is entering or leaving the
            // medium.
            etaScale *= (dot(wo, surfelY.shadingNormal) > 0.0) ? (eta_for_RR * eta_for_RR) : 1.0 / (eta_for_RR * eta_for_RR);

            // Possibly terminate the path with Russian roulette.
            // Factor out radiance scaling due to refraction in rrBeta.
            float3 rrBeta = beta * etaScale;
            float maxRR = max(rrBeta.x, max(rrBeta.y, rrBeta.z));
            if (maxRR < 1.0 && i > 3) {
                float q = max(0.05, 1.0 - maxRR);
                if (rnd(surfelY.seed) < q) {
                    break;
                }
                beta = beta / (1.0 - q);
            }
        }
        else {
            L = L + beta * skyColor(surfelY.wi);
            break;
        }
    }

    return L;
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    unsigned int p0, p1;
    Surfel surfelY;
    packPointer(&surfelY, p0, p1);

    surfelY.seed = tea<4>(idx.y * params.image_width + idx.x, 250);

    float3 L = make_float3(0.0, 0.0, 0.0);

    float3 ray_origin;
    float3 ray_dir;

    int NSAMPLES = 50;

    float3 uv = make_float3(
        static_cast<float>(idx.x) / static_cast<float>(dim.x) - 0.5,
        static_cast<float>(idx.y) / static_cast<float>(dim.y) - 0.5,
        0.0f
    );

    int nsamps = int(1.0 / (1.0 + 0.5 * sqrtf(dot(uv, uv))) * float(NSAMPLES));
    float oneOverSPP = 1.0 / float(nsamps);
    float strataSize = oneOverSPP;
    for (int i = 0; i < nsamps; i++) {
        computeRay(idx, dim, ray_origin, ray_dir, surfelY.seed);
        L = L + L_i(surfelY, ray_origin, ray_dir, p0, p1);
    }

    L.x = L.x * oneOverSPP;
    L.y = L.y * oneOverSPP;
    L.z = L.z * oneOverSPP;

    params.image[idx.y * params.image_width + idx.x] = make_color(L);
}

extern "C" __global__ void __closesthit__ch() {
    const HitGroupData& sbtData = *(const HitGroupData*)optixGetSbtDataPointer();
    
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // Which triangle did we intercept
    const int prim_idx = optixGetPrimitiveIndex();
        
    // This is how we get the vertices of the intersected triangle
    const int3 ind = sbtData.indices[prim_idx];
    const float3 v0 = sbtData.vertices[ind.x];
    const float3 v1 = sbtData.vertices[ind.y];
    const float3 v2 = sbtData.vertices[ind.z];

    // standard trick for getting the normal of the intersected triangle
    const float3 nor_0 = normalize(cross(v1 - v0, v2 - v0));
    
    const float3 ray_dir = optixGetWorldRayDirection();

    // Make sure the normal is not pointing inward
    // code from RTIOW
    const float3 nor = set_face_normal(ray_dir, nor_0);

    Surfel* surfelY = unpackSurfel();

    surfelY->hit = true;
    float t = optixGetRayTmax();
    surfelY->t = t;
    surfelY->position = optixGetWorldRayOrigin() + t * ray_dir;
    surfelY->shadingNormal = nor;
    surfelY->albedo = sbtData.albedo;
}

extern "C" __global__ void __miss__ms() {
    const float3 ray_dir = optixGetWorldRayDirection();

    Surfel* surfelX = unpackSurfel();
    
    surfelX->hit = false;
    surfelX->wi = ray_dir;
}