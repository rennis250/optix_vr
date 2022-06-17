﻿#include "rob_optix.h"
#include "SDL.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#include <iostream>
#include <fstream>

const unsigned int width = 256;
const unsigned int height = 256;

int main(int argc, char* argv[]) {
    SDL_Init(SDL_INIT_EVERYTHING);
    SDL_Window* window = SDL_CreateWindow("SDL2 Test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    auto optState = rob::OptixState<uchar4>();

    const std::vector<float3> vertices = {
        { -0.5f, -0.5f, 0.0f },
        {  0.5f, -0.5f, 0.0f },
        {  0.0f,  0.5f, 0.0f }
    };

    optState.build_shape(vertices);
    optState.setup_gas();
    optState.create_module();

    optState.generate_program_group("__raygen__rg", OPTIX_PROGRAM_GROUP_KIND_RAYGEN);
    optState.generate_program_group("__miss__ms", OPTIX_PROGRAM_GROUP_KIND_MISS);
    optState.generate_program_group("__closesthit__ch", OPTIX_PROGRAM_GROUP_KIND_HITGROUP);

    optState.create_pipeline();

    MissSbtRecord ms_sbt = {};
    ms_sbt.data.bg_color = { 0.3f, 0.1f, 0.2f };
    optState.allocate_record(ms_sbt);

    RayGenSbtRecord rg_sbt = {};
    optState.allocate_record(rg_sbt);
    
    HitGroupSbtRecord hg_sbt = {};
    optState.allocate_record(hg_sbt);

    optState.build_sbt();

    auto d_image = rob::CudaBuffer<uchar4>(width, height);
    
    // Populate the per-launch params
    Params params;
    params.image = d_image.get_device_ptr();
    params.image_width = width;
    params.image_height = height;
    params.cam_eye = make_float3(0.0f, 0.0f, 2.0f);
    params.cam_u = make_float3(1.0f, 0.0f, 0.0f);
    params.cam_v = make_float3(0.0f, 1.0f, 0.0f);
    params.cam_w = make_float3(0.0f, 0.0f, -2.0f);

    optState.upload_params(params);
    optState.render();

    d_image.download();
    std::vector<uchar4> host_pixels = d_image.get_host_data();

    // Output FB as Image
    std::ofstream MyFile("test.ppm");
    MyFile << "P3\n" << width << " " << height << "\n255\n";
    for (int j = (int)height - 1; j >= 0; j--) {
        for (int i = 0; i < (int)width; i++) {
            size_t pixel_index = j * (size_t)width + i;
            
            int ir = static_cast<int>(host_pixels[pixel_index].x);
            int ig = static_cast<int>(host_pixels[pixel_index].y);
            int ib = static_cast<int>(host_pixels[pixel_index].z);

            MyFile << ir << " " << ig << " " << ib << "\n";
        }
    }
    MyFile.close();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}