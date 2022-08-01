#include "rob_optix.h"
#include "rob_sdl.h"
#include "rob_opengl.h"
#include "rob_mesh.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#include <iostream>
#include <fstream>

#include "exceptions.h"

const unsigned int width = 256;
const unsigned int height = 256;

int main(int argc, char* argv[]) {
	auto sdlApp = rob::SDLApp(width, height);
	auto glState = rob::GLState();
	auto optState = rob::OptixState<uchar4>();

	glState.buildPBO(width, height);
	glState.createTexture();
	glState.buildVAO();
	glState.buildVBOandEBO();
	glState.buildShaderProgram();

	cudaGraphicsResource* m_cuda_gfx_resource = nullptr;
	CUDA_CHECK(
		cudaGraphicsGLRegisterBuffer(
			&m_cuda_gfx_resource,
			glState.m_pbo,
			cudaGraphicsMapFlagsWriteDiscard
		)
	);

	std::vector<uchar4> m_host_pixels;
	m_host_pixels.resize(width * height);

	rob::Model model;
	model.load();

	std::vector<rob::Mesh> meshes(2);
	
	float3 center = make_float3(0.5, 0.5, -0.5);
	float3 size = make_float3(0.5, 1.0, 1.0);
	float3 color = make_float3(0.8, 0.1, 0.1);
	int mat = 0;
	meshes[0].make_cube(center, size, color, mat);

	center = make_float3(0.0, 0.0, 0.0);
	size = make_float3(100.0, 0.1, 100.0);
	color = make_float3(0.5, 0.5, 0.5);
	mat = 1;
	meshes[1].make_cube(center, size, color, mat);

	optState.build_shape(model.get_meshes());
	// optState.build_shape(meshes);
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

	optState.allocate_hg_records();

	optState.build_sbt();

	// Populate the per-launch params
	Params params;
	params.image_width = width;
	params.image_height = height;
	params.cam_eye = make_float3(0.0f, 1.5f, 11.0f);
	// params.cam_eye = make_float3(-1293.07f, 154.681f, -0.7304f);
	params.cam_u = make_float3(1.0f, 0.0f, 0.0f);
	params.cam_v = make_float3(0.0f, 1.0f, 0.0f);
	params.cam_w = make_float3(0.0f, 0.0f, -2.0f);

	uchar4* gl_image = nullptr;
	size_t buffer_size = 0u;

	try {
		CUDA_CHECK(
			cudaGraphicsMapResources(1, &m_cuda_gfx_resource, 0)
		);
		CUDA_CHECK(
			cudaGraphicsResourceGetMappedPointer(
				reinterpret_cast<void**>(&gl_image),
				&buffer_size,
				m_cuda_gfx_resource
			)
		);
	}
	catch (std::exception& e)
	{
		std::cerr << "Optix error: " << e.what() << std::endl;
	}
	params.image = gl_image;

	int x, y;
	while (1) {
		sdlApp.clearScreen();
		sdlApp.registerInput(x, y);

		params.cam_eye = make_float3(
			static_cast<float>(x)/static_cast<float>(width) + 0.5,
			static_cast<float>(y)/static_cast<float>(height) + 0.5,
			6.0f
		);
		
		optState.upload_params(params);
		optState.render();

		glState.draw();

		sdlApp.drawScene();

		SDL_Delay(16);
	}

	try {
		CUDA_CHECK(
			cudaGraphicsUnmapResources(
				1,
				&m_cuda_gfx_resource,
				0
			)
		);
	}
	catch (std::exception& e)
	{
		std::cerr << "Optix error: " << e.what() << std::endl;
	}

	return 0;
}