#include <optix.h>
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <iostream>
#include <array>

#include "launch_params.h"
#include "exceptions.h"
#include "ptx_loader.h"

// SBT record with an appropriately aligned and sized data block
template<typename T> struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

char olog[2048];

int main() {
    // Initialize CUDA with a no-op call to the the CUDA runtime API
    CUDA_CHECK(cudaFree(0));

    // Initialize the OptiX API, loading all API entry points 
    OPTIX_CHECK(optixInit());

    // Specify options for this context. We will use the default options.
    OptixDeviceContextOptions options = {};

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0; // NULL means take the current active context

    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    // Specify options for the build. We use default options for simplicity.
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Triangle build input: simple list of three vertices
    const std::array<float3, 3> vertices = {
        {
            { -0.5f, -0.5f, 0.0f },
            {  0.5f, -0.5f, 0.0f },
            {  0.0f,  0.5f, 0.0f }
        }
    };

    // Allocate and copy device memory for our input triangle vertices
    const size_t vertices_size = sizeof(float3) * vertices.size();
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice));

    // Populate the build input struct with our triangle data as well as
    // information about the sizes and types of our data
    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = vertices.size();
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    // Query OptiX for the memory requirements for our GAS 
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,         // The device context we are using
        &accel_options,
        &triangle_input, // Describes our geometry
        1,               // Number of build inputs, could have multiple
        &gas_buffer_sizes));

    // Allocate device memory for the scratch space buffer as well
    // as the GAS itself
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes));
    CUdeviceptr d_gas_output_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer),
        gas_buffer_sizes.outputSizeInBytes));

    // Now build the GAS
    OptixTraversableHandle gas_handle{ 0 };
    OPTIX_CHECK(optixAccelBuild(
        context,
        0,           // CUDA stream
        &accel_options,
        &triangle_input,
        1,           // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle, // Output handle to the struct
        nullptr,     // emitted property list
        0));         // num emitted properties

    // We can now free scratch space used during the build
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));

    // Default options for our module.
    OptixModuleCompileOptions module_compile_options = {};

    // Pipeline options must be consistent for all modules used in a
    // single pipeline
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;

    // This option is important to ensure we compile code which is optimal
    // for our scene hierarchy. We use a single GAS � no instancing or
    // multi-level hierarchies
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    // Our device code uses 3 payload registers (r,g,b output value)
    pipeline_compile_options.numPayloadValues = 3;

    // This is the name of the param struct variable in our device code
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t sizeof_olog = sizeof(olog);
    size_t      inputSize = 0;
    const char* input = readPTX(&inputSize);

    OptixModule module = nullptr; // The output module
    OPTIX_CHECK(optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        input,
        inputSize,
        olog,
        &sizeof_olog,
        &module
    ));

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    sizeof_olog = sizeof(olog);
    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        olog,
        &sizeof_olog,
        &raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_olog = sizeof(olog);
    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        olog,
        &sizeof_olog,
        &miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    // We could also specify an any-hit and/or intersection program here
    sizeof_olog = sizeof(olog);
    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        olog,
        &sizeof_olog,
        &hitgroup_prog_group));

    OptixProgramGroup program_groups[] =
    {
        raygen_prog_group,
        miss_prog_group,
        hitgroup_prog_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    sizeof_olog = sizeof(olog);

    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        olog,
        &sizeof_olog,
        &pipeline));

    // Allocate the miss record on the device 
    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));

    // Populate host-side copy of the record with header and data
    MissSbtRecord ms_sbt;
    ms_sbt.data.bg_color = { 0.3f, 0.1f, 0.2f };
    optixSbtRecordPackHeader(miss_prog_group, &ms_sbt);

    // Now copy our host record to the device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr rg_record;
    size_t rg_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rg_record), rg_record_size));

    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));

    // Now copy our host record to the device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(rg_record),
        &rg_sbt,
        rg_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr hg_record;
    size_t hg_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hg_record), hg_record_size));

    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));

    // Now copy our host record to the device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(hg_record),
        &hg_sbt,
        hg_record_size,
        cudaMemcpyHostToDevice));

    // The shader binding table struct we will populate
    OptixShaderBindingTable sbt = {};

    // Finally we specify how many records and how they are packed in memory
    sbt.raygenRecord = rg_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = hg_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1;

    unsigned int width = 256;
    unsigned int height = 256;

    uchar4* image_data = new uchar4[(size_t)width * (size_t)height];

    // Populate the per-launch params
    Params params;
    params.image = image_data;
    params.image_width = width;
    params.image_height = height;
    params.cam_eye = float3(0.0f, 0.0f, 2.0f);
    params.cam_u = float3(1.0f, 0.0f, 0.0f);
    params.cam_v = float3(0.0f, 1.0f, 0.0f);
    params.cam_w = float3(0.0f, 0.0f, -2.0f);
    params.handle = gas_handle;

    // Transfer params to the device
    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param),
        &params, sizeof(params),
        cudaMemcpyHostToDevice));

    // Launch now, passing in our pipeline, launch params, and SBT
    OPTIX_CHECK(optixLaunch(pipeline,
        0,   // Default CUDA stream
        d_param,
        sizeof(Params),
        &sbt,
        width,
        height,
        1)); // depth

    width = 2;
    height = 2;
    // Output FB as Image
    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * (size_t)width + i;
            float r = static_cast<float>(image_data[pixel_index].x);
            float g = static_cast<float>(image_data[pixel_index].y);
            float b = static_cast<float>(image_data[pixel_index].z);
            int ir = int(r);
            int ig = int(g);
            int ib = int(b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    return 0;
}