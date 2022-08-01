#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "cuda_buffer.h"
#include "ptx_loader.h"
#include "exceptions.h"
#include "launch_params.h"
#include "sbt_record.h"
#include "rob_mesh.h"

#include <vector>
#include <iostream>

namespace rob {
	template<typename T>
	class OptixState {
	public:
		OptixState();
		~OptixState();

		void build_shape(std::vector<Mesh> meshes);
		void setup_gas();
		void create_module();
		void generate_program_group(const char* func_name, OptixProgramGroupKind prog_kind);
		void create_pipeline();
		void allocate_record(RayGenSbtRecord& sbt);
		void allocate_record(MissSbtRecord& sbt);
		// void allocate_record(HitGroupSbtRecord& sbt);
		void allocate_hg_records();
		void build_sbt();
		void upload_params(Params params);
		void render();

	private:
		CUcontext m_cuCtx = 0;
		OptixDeviceContext m_optCtx = nullptr;

		OptixModule m_module = nullptr;
		OptixPipeline m_pipeline = nullptr;
		OptixShaderBindingTable m_sbt = {};

		OptixDeviceContextOptions m_options = {};
		OptixAccelBuildOptions m_accel_options = {};
		OptixModuleCompileOptions m_module_compile_options = {};
		OptixPipelineCompileOptions m_pipeline_compile_options = {};
		OptixProgramGroupOptions m_program_group_options = {};
		OptixPipelineLinkOptions m_pipeline_link_options = {};

		std::vector<OptixBuildInput> m_build_inputs;
		std::vector<uint32_t> m_build_input_flags;

		OptixAccelBufferSizes m_gas_buffer_sizes = {};
		OptixTraversableHandle m_gas_handle = 0;

		CUdeviceptr m_temp_buffer_gas = 0;
		CUdeviceptr m_gas_output_buffer = 0;
		CUdeviceptr m_d_param = 0;
		std::vector<CUdeviceptr> m_d_vertices;
		std::vector<CUdeviceptr> m_d_indices;
		std::vector<CUdeviceptr> m_d_mats;

		OptixProgramGroup m_program_groups[3] = { };
		OptixProgramGroup m_raygen_prog_group;
		OptixProgramGroup m_miss_prog_group;
		OptixProgramGroup m_hitgroup_prog_group;

		CUdeviceptr m_miss_record;
		CUdeviceptr m_rg_record;
		CUdeviceptr m_hg_record;

		std::vector<HitGroupSbtRecord> m_hg_sbts;

		Params m_params;

		std::vector<Mesh> m_meshes;
	};

	template<typename T>
	OptixState<T>::OptixState() {
		try {
			CUDA_CHECK(cudaFree(0));
			OPTIX_CHECK(optixInit());
			OPTIX_CHECK(optixDeviceContextCreate(m_cuCtx, &m_options, &m_optCtx));

			m_accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
			m_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}
	}

	template<typename T>
	OptixState<T>::~OptixState() {
		try {
			OPTIX_CHECK(optixDeviceContextDestroy(m_optCtx));
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}
	}

	template<typename T>
	void OptixState<T>::build_shape(std::vector<Mesh> meshes) {
		m_meshes = meshes;

		m_d_vertices.resize(m_meshes.size());
		m_d_indices.resize(m_meshes.size());
		m_build_input_flags.resize(m_meshes.size());
		m_build_inputs.resize(m_meshes.size());

		try {
			for (int meshID = 0; meshID < m_meshes.size(); meshID++) {
				const size_t vertices_size_in_bytes = m_meshes[meshID].get_vertices().size() * sizeof(float3);
				CUDA_CHECK(
					cudaMalloc(
						reinterpret_cast<void**>(&m_d_vertices[meshID]),
						vertices_size_in_bytes
					)
				);

				CUDA_CHECK(
					cudaMemcpy(
						reinterpret_cast<void*>(m_d_vertices[meshID]),
						m_meshes[meshID].get_vertices().data(),
						vertices_size_in_bytes,
						cudaMemcpyHostToDevice
					)
				);

				const size_t vert_indices_size_in_bytes = m_meshes[meshID].get_indices().size() * sizeof(int3);
				CUDA_CHECK(
					cudaMalloc(
						reinterpret_cast<void**>(&m_d_indices[meshID]),
						vert_indices_size_in_bytes
					)
				);

				CUDA_CHECK(
					cudaMemcpy(
						reinterpret_cast<void*>(m_d_indices[meshID]),
						m_meshes[meshID].get_indices().data(),
						vert_indices_size_in_bytes,
						cudaMemcpyHostToDevice
					)
				);
			}
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		for (int meshID = 0; meshID < m_meshes.size(); meshID++) {
			m_build_input_flags[meshID] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

			m_build_inputs[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			m_build_inputs[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			m_build_inputs[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
			m_build_inputs[meshID].triangleArray.numVertices = static_cast<uint32_t>(m_meshes[meshID].get_vertices().size());
			m_build_inputs[meshID].triangleArray.vertexBuffers = &m_d_vertices[meshID];
			m_build_inputs[meshID].triangleArray.flags = &m_build_input_flags[meshID];
			m_build_inputs[meshID].triangleArray.numSbtRecords = 1;
			m_build_inputs[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
			m_build_inputs[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
			m_build_inputs[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;

			m_build_inputs[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			m_build_inputs[meshID].triangleArray.indexStrideInBytes = sizeof(int3);
			m_build_inputs[meshID].triangleArray.numIndexTriplets = (int)m_meshes[meshID].get_indices().size();
			m_build_inputs[meshID].triangleArray.indexBuffer = m_d_indices[meshID];
		}

		return;
	}

	template<typename T>
	void OptixState<T>::setup_gas() {
		try {
			OPTIX_CHECK(
				optixAccelComputeMemoryUsage(
					m_optCtx,
					&m_accel_options,
					m_build_inputs.data(),
					m_build_inputs.size(),  // num_build_inputs
					&m_gas_buffer_sizes
				)
			);

			CUDA_CHECK(
				cudaMalloc(
					reinterpret_cast<void**>(&m_temp_buffer_gas),
					m_gas_buffer_sizes.tempSizeInBytes
				)
			);

			CUDA_CHECK(
				cudaMalloc(
					reinterpret_cast<void**>(&m_gas_output_buffer),
					m_gas_buffer_sizes.outputSizeInBytes
				)
			);

			OPTIX_CHECK(
				optixAccelBuild(
					m_optCtx,
					0,                                  // CUDA stream
					&m_accel_options,
					m_build_inputs.data(),
					m_build_inputs.size(),                                  // num build inputs
					m_temp_buffer_gas,
					m_gas_buffer_sizes.tempSizeInBytes,
					m_gas_output_buffer,
					m_gas_buffer_sizes.outputSizeInBytes,
					&m_gas_handle,
					nullptr,                      // emitted property list
					0                                   // num emitted properties
				)
			);

			CUDA_SYNC_CHECK();

			CUDA_CHECK(
				cudaFree(
					reinterpret_cast<void*>(m_temp_buffer_gas)
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}

	template<typename T>
	void OptixState<T>::create_module() {
		try {
			m_pipeline_compile_options.usesMotionBlur = false;
			m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
			m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

			m_pipeline_compile_options.numPayloadValues = 2;
			m_pipeline_compile_options.numAttributeValues = 2;

			char olog[2048];
			size_t sizeof_olog = sizeof(olog);
			size_t inputSize = 0;
			const char* input = readPTX(&inputSize);

			OPTIX_CHECK(
				optixModuleCreateFromPTX(
					m_optCtx,
					&m_module_compile_options,
					&m_pipeline_compile_options,
					input,
					inputSize,
					olog,
					&sizeof_olog,
					&m_module
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}

	template<typename T>
	void OptixState<T>::generate_program_group(const char* func_name, OptixProgramGroupKind prog_kind) {
		try {
			OptixProgramGroup prog_group = nullptr;

			OptixProgramGroupDesc prog_group_desc = {};
			prog_group_desc.kind = prog_kind;

			switch (prog_kind)
			{
			case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
				prog_group_desc.raygen.module = m_module;
				prog_group_desc.raygen.entryFunctionName = func_name;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_MISS:
				prog_group_desc.miss.module = m_module;
				prog_group_desc.miss.entryFunctionName = func_name;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
				break;
			case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
				prog_group_desc.hitgroup.moduleCH = m_module;
				prog_group_desc.hitgroup.entryFunctionNameCH = func_name;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
				break;
			default:
				break;
			}

			char olog[2048];
			size_t sizeof_olog = sizeof(olog);

			OPTIX_CHECK(
				optixProgramGroupCreate(
					m_optCtx,
					&prog_group_desc,
					1, // num program groups
					&m_program_group_options,
					olog,
					&sizeof_olog,
					&prog_group
				)
			);

			// m_program_groups.push_back(prog_group);

			switch (prog_kind)
			{
			case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
				m_raygen_prog_group = prog_group;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_MISS:
				m_miss_prog_group = prog_group;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
				break;
			case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
				m_hitgroup_prog_group = prog_group;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
				break;
			default:
				break;
			}
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}

	template<typename T>
	void OptixState<T>::create_pipeline() {
		try {
			m_pipeline_link_options.maxTraceDepth = 2;
			m_pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

			char olog[2048];
			size_t sizeof_olog = sizeof(olog);

			m_program_groups[0] = m_raygen_prog_group;
			m_program_groups[1] = m_miss_prog_group;
			m_program_groups[2] = m_hitgroup_prog_group;

			OPTIX_CHECK(
				optixPipelineCreate(
					m_optCtx,
					&m_pipeline_compile_options,
					&m_pipeline_link_options,
					&m_program_groups[0],
					sizeof(m_program_groups) / sizeof(m_program_groups[0]),
					olog,
					&sizeof_olog,
					&m_pipeline
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}

	template <typename T>
	void OptixState<T>::allocate_record(MissSbtRecord& sbt) {
		try {
			size_t record_size = sizeof(MissSbtRecord);
			CUDA_CHECK(
				cudaMalloc(
					reinterpret_cast<void**>(&m_miss_record),
					record_size
				)
			);

			OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_prog_group, &sbt));

			CUDA_CHECK(
				cudaMemcpy(
					reinterpret_cast<void*>(m_miss_record),
					&sbt,
					record_size,
					cudaMemcpyHostToDevice
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}

	template <typename T>
	void OptixState<T>::allocate_record(RayGenSbtRecord& sbt) {
		try {
			size_t record_size = sizeof(RayGenSbtRecord);
			CUDA_CHECK(
				cudaMalloc(
					reinterpret_cast<void**>(&m_rg_record),
					record_size
				)
			);

			OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &sbt));

			CUDA_CHECK(
				cudaMemcpy(
					reinterpret_cast<void*>(m_rg_record),
					&sbt,
					record_size,
					cudaMemcpyHostToDevice
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}

	template <typename T>
	void OptixState<T>::allocate_hg_records() {
		try {
			m_hg_sbts.resize(m_meshes.size());
			for (int meshID = 0; meshID < m_meshes.size(); meshID++) {
				HitGroupSbtRecord sbt = {};
				OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroup_prog_group, &sbt));

				sbt.data.albedo = m_meshes[meshID].get_albedo();

				std::cout << sbt.data.albedo.x << ", " << sbt.data.albedo.y << ", " << sbt.data.albedo.z << std::endl;

				sbt.data.vertices = (float3*)m_d_vertices[meshID];
				sbt.data.indices = (int3*)m_d_indices[meshID];

				m_hg_sbts[meshID] = sbt;
			}

			size_t record_size = sizeof(HitGroupSbtRecord) * m_hg_sbts.size();
			CUDA_CHECK(
				cudaMalloc(
					reinterpret_cast<void**>(&m_hg_record),
					record_size
				)
			);

			CUDA_CHECK(
				cudaMemcpy(
					reinterpret_cast<void*>(m_hg_record),
					m_hg_sbts.data(),
					record_size,
					cudaMemcpyHostToDevice
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}

	template <typename T>
	void OptixState<T>::build_sbt() {
		m_sbt.raygenRecord = m_rg_record;
		m_sbt.missRecordBase = m_miss_record;
		m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
		m_sbt.missRecordCount = 1;
		m_sbt.hitgroupRecordBase = m_hg_record;
		m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
		m_sbt.hitgroupRecordCount = m_hg_sbts.size();

		return;
	}

	template <typename T>
	void OptixState<T>::upload_params(Params params) {
		try {
			m_params = params;
			m_params.handle = m_gas_handle;

			CUDA_CHECK(
				cudaMalloc(
					reinterpret_cast<void**>(&m_d_param),
					sizeof(Params)
				)
			);

			CUDA_CHECK(
				cudaMemcpy(
					reinterpret_cast<void*>(m_d_param),
					&m_params,
					sizeof(m_params),
					cudaMemcpyHostToDevice
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}

	template <typename T>
	void OptixState<T>::render() {
		try {
			OPTIX_CHECK(
				optixLaunch(
					m_pipeline,
					0,   // Default CUDA stream
					m_d_param,
					sizeof(Params),
					&m_sbt,
					m_params.image_width,
					m_params.image_height,
					1    // depth
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "Optix error: " << e.what() << std::endl;
		}

		return;
	}
}