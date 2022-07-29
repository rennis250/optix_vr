#include <optix.h>
#include <cuda_runtime.h>
#include <optix_device.h>

#include "launch_params.h"
#include "vec.h"
#include "surfel.h"
#include "color.h"
#include "materials.h"

extern "C" {
    __constant__ Params params;
}

__device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
    float3 U = params.cam_u;
    float3 V = params.cam_v;
    float3 W = params.cam_w;

    // Normalizing coordinates to [-1.0, 1.0]
    float3 d = 2.0f * make_float3(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y),
        0.0f
    ) - 1.0f;

    origin = params.cam_eye;
    direction = normalize(d.x * U + d.y * V + W);

    return;
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // this puts the pointer value for the surfel
    // into two uints so that they can go into
    // the payload portion of optixTrace,
    // where they are reconstructed into the
    // pointer itself (tricky, but cool!)
    unsigned int p0, p1;
    Surfel surfelY;
    packPointer(&surfelY, p0, p1);

    surfelY.seed = tea<4>(idx.y * params.image_width + idx.x, 250);

    // Map our launch idx to a screen location and create a ray from 
    // the camera location through the screen
    float3 X;
    computeRay(idx, dim, X, surfelY.wi);
    float3 X_for_vignette = X;

    float3 L = make_float3(0.0, 0.0, 0.0);
    float3 beta = make_float3(1.0, 1.0, 1.0);
    float pdf = 0.0;
    float3 fr, wo;
    float eta_for_RR = 0.0;
    float etaScale = 0.0;

    int nbounces = 10;
    for (int i = 0; i < nbounces; i++) {
        // Trace the ray against our scene hierarchy.
        // optixTrace is basically the following function
        // from the WebXR version: findFirstIntersection
        optixTrace(
            params.handle,
            X,
            surfelY.wi,
            0.0f,   // Min intersection distance
            1e16f,  // Max intersection distance
            0.0f,   // ray-time -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,      // SBT offset -- See SBT discussion
            0,      // SBT stride -- See SBT discussion 
            0,      // missSBTIndex -- See SBT discussion
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
        } else {
            L = L + beta * skyColor(surfelY.wi);
            break;
        }
    }

    // float col = spect_to_rgb(L);
    float3 col = L;
    // col *= oneOverSPP;
    col = clamp(col, 0.0, 1000.0); // prevent NaN and Inf

    // vignetting
    // float3 p = X_for_vignette;
    // col = col * (0.5 + 0.5 * powf(16.0 * p.x * p.y * (1.0 - p.x) * (1.0 - p.y), 0.3));

    // Record results in our output raster
    // params.image[idx.y * params.image_width + idx.x] = make_color(reinhardTonemap(col, 0.8, 0.1));
    params.image[idx.y * params.image_width + idx.x] = make_color(col);
}

extern "C" __global__ void __closesthit__ch() {
    const HitGroupData& sbtData
        = *(const HitGroupData*)optixGetSbtDataPointer();

    // When built-in triangle intersection is used, a number of fundamental 
    // attributes are provided by the OptiX API, including barycentric 
    // coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();

    // What is the direction of the ray at intersection
    const float3 ray_dir = optixGetWorldRayDirection();

    // Which triangle did we intercept
    const int prim_idx = optixGetPrimitiveIndex();
        
    // This is how we get the vertices of the intersected triangle
    const int3 ind = sbtData.indices[prim_idx];
    const float3 v0 = sbtData.vertices[ind.x];
    const float3 v1 = sbtData.vertices[ind.y];
    const float3 v2 = sbtData.vertices[ind.z];

    // standard trick for getting the normal of the intersected triangle
    const float3 nor_0 = normalize(cross(v1 - v0, v2 - v0));

    // Make sure the normal is not pointing inward
    // code from RTIOW
    const float3 nor = set_face_normal(ray_dir, nor_0);

    Surfel* surfelX = unpackSurfel();

    surfelX->hit = true;
    surfelX->wi = ray_dir;
    float t = optixGetRayTmax();
    surfelX->t = t;
    surfelX->position = optixGetWorldRayOrigin() + t * ray_dir;
    surfelX->shadingNormal = nor;

    // surfelX->mat = getMaterial(TEXTURE, surfelX->position, surfelX->shadingNormal);
    surfelX->albedo = sbtData.albedo;
}

extern "C" __global__ void __miss__ms() {
    // MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    
    const float3 ray_dir = optixGetWorldRayDirection();

    Surfel* surfelX = unpackSurfel();
    
    surfelX->hit = false;
    surfelX->wi = ray_dir;
}