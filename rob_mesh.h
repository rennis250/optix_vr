#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include <vector>
#include "vec.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace rob {
	class Mesh {
	public:
		void make_cube(float3 center, float3 size, float3 albedo, int mat);
		std::vector<float3> get_vertices();
		std::vector<int3> get_indices();
		float3 get_albedo();
		int get_mats();

		void add_vertex(float3 v);
		void add_index(int3 i);
		void set_albedo(float3 a);

	private:
		std::vector<float3> m_vertices;
		std::vector<int3> m_indices;
		float3 m_albedo;
		int m_mats;
	};

	void Mesh::add_vertex(float3 v) {
		m_vertices.push_back(v);
	}

	void Mesh::add_index(int3 i) {
		m_indices.push_back(i);
	}

	void Mesh::set_albedo(float3 a) {
		m_albedo = a;
	}

	std::vector<float3> Mesh::get_vertices() {
		return m_vertices;
	}

	std::vector<int3> Mesh::get_indices() {
		return m_indices;
	}

	float3 Mesh::get_albedo() {
		return m_albedo;
	}

	int Mesh::get_mats() {
		return m_mats;
	}

	void Mesh::make_cube(float3 center, float3 size, float3 albedo, int mat) {
		m_albedo = albedo;
		m_mats = mat;

		float3 p = center - 0.5f * size;
		float3 coor = p + size;

		int lastVertixInd = m_vertices.size();
		m_vertices.push_back(make_float3(p.x,    p.y,    p.z   )); // 0 - origin
		m_vertices.push_back(make_float3(coor.x, p.y,    p.z   )); // 1 - pure x
		m_vertices.push_back(make_float3(p.x,    coor.y, p.z   )); // 2 - pure y
		m_vertices.push_back(make_float3(coor.x, coor.y, p.z   )); // 3 - diag x/y
		m_vertices.push_back(make_float3(p.x,    p.y,    coor.z)); // 4 - pure z
		m_vertices.push_back(make_float3(coor.x, p.y,    coor.z)); // 5 - diag x/z
		m_vertices.push_back(make_float3(p.x,    coor.y, coor.z)); // 6 - diag y/z
		m_vertices.push_back(make_float3(coor.x, coor.y, coor.z)); // 7 - furthest tip

		int indices[] = {
			0, 1, 2,   1, 2, 3, // x/y wall
			0, 1, 4,   1, 4, 5, // y/z wall
			2, 6, 7,   2, 7, 3, // top wall
			0, 4, 5,   0, 5, 1, // bot wall
			4, 5, 6,   6, 5, 7, // y/z far wall
			1, 5, 7,   1, 7, 3  // x/y far wall
		};
		for (int i = 0; i < 12; i++) {
			int3 is = make_int3(lastVertixInd + indices[3 * i + 0],
				lastVertixInd + indices[3 * i + 1],
				lastVertixInd + indices[3 * i + 2]);
			m_indices.push_back(is);
		}
	}

	class Model {
	public:
		void load();
		std::vector<Mesh> get_meshes();

	private:
		std::vector<Mesh> m_meshes;
	};

	std::vector<Mesh> Model::get_meshes() {
		return m_meshes;
	}

	void Model::load() {
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "./models/sponza.obj", "./models", true)) {
			throw std::runtime_error(warn + err);
		}

		if (materials.empty())
			throw std::runtime_error("could not parse materials ...");

		std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;

		for (const auto& shape : shapes) {
			rob::Mesh mesh;

			for (const auto& index : shape.mesh.indices) {
				float3 vertex = make_float3(0.0, 0.0, 0.0);

				vertex.x = attrib.vertices[3 * index.vertex_index + 0];
				vertex.y = attrib.vertices[3 * index.vertex_index + 1];
				vertex.z = attrib.vertices[3 * index.vertex_index + 2];

				mesh.add_vertex(vertex);

				mesh.set_albedo(
					make_float3(
						static_cast<float>(rand() % 255)/255.0,
						static_cast<float>(rand() % 255) / 255.0,
						static_cast<float>(rand() % 255) / 255.0
					)
				);

				mesh.add_index(
					make_int3(
						mesh.get_indices().size() + 0,
						mesh.get_indices().size() + 1,
						mesh.get_indices().size() + 2
					)
				);
			}

			m_meshes.push_back(mesh);
		}
	}
}