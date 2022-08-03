#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include <vector>
#include <set>

#include "vec.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace std {
	inline bool operator<(const tinyobj::index_t& a,
		const tinyobj::index_t& b)
	{
		if (a.vertex_index < b.vertex_index) return true;
		if (a.vertex_index > b.vertex_index) return false;

		if (a.normal_index < b.normal_index) return true;
		if (a.normal_index > b.normal_index) return false;

		if (a.texcoord_index < b.texcoord_index) return true;
		if (a.texcoord_index > b.texcoord_index) return false;

		return false;
	}
}

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
		void set_mat(int);

		int add_obj_vertex(tinyobj::attrib_t attrib, tinyobj::index_t index, std::map<tinyobj::index_t, int> knownVertices);

	private:
		std::vector<float3> m_vertices;
		std::vector<int3> m_indices;
		float3 m_albedo;
		int m_mats;
	};

	void Mesh::add_vertex(float3 v) {
		m_vertices.push_back(v);
	}

	void Mesh::set_mat(int m) {
		m_mats = m;
	}

	int Mesh::add_obj_vertex(tinyobj::attrib_t attrib, tinyobj::index_t index, std::map<tinyobj::index_t, int> knownVertices) {
		if (knownVertices.find(index) != knownVertices.end()) {
			return knownVertices[index];
		}

		int newID = m_vertices.size();
		knownVertices[index] = newID;

		float3 vertex = make_float3(0.0, 0.0, 0.0);

		vertex.x = attrib.vertices[3 * index.vertex_index + 0];
		vertex.y = attrib.vertices[3 * index.vertex_index + 1];
		vertex.z = attrib.vertices[3 * index.vertex_index + 2];

		m_vertices.push_back(vertex);

		return newID;
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

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "./models/test_rob.obj", "./models", true)) {
			throw std::runtime_error(warn + err);
		}

		if (materials.empty())
			throw std::runtime_error("could not parse materials ...");

		std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;

		for (const auto& shape : shapes) {
			std::map<tinyobj::index_t, int> knownVertices;

			std::set<int> materialIDs;
			for (auto faceMatID : shape.mesh.material_ids) {
				materialIDs.insert(faceMatID);
			}

			for (int materialID : materialIDs) {
				rob::Mesh mesh;

				mesh.set_mat(materialID);

				for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
					if (shape.mesh.material_ids[faceID] != materialID) {
						continue;
					}

					tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
					tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
					tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

					int3 idx = make_int3(
						mesh.add_obj_vertex(attrib, idx0, knownVertices),
						mesh.add_obj_vertex(attrib, idx1, knownVertices),
						mesh.add_obj_vertex(attrib, idx2, knownVertices)
					);

					mesh.add_index(idx);

					mesh.set_albedo(
						make_float3(
							materials[materialID].diffuse[0],
							materials[materialID].diffuse[1],
							materials[materialID].diffuse[2]
						)
					);
				}

				m_meshes.push_back(mesh);
			}
		}
	}
}