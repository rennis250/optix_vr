#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include <vector>
#include "vec.h"

namespace rob {
	class Mesh {
	public:
		void make_cube(float3 center, float3 size, float3 albedo, int mat);
		std::vector<float3> get_vertices();
		std::vector<int3> get_indices();
		float3 get_albedo();
		int get_mats();

	private:
		std::vector<float3> m_vertices;
		std::vector<int3> m_indices;
		float3 m_albedo;
		int m_mats;
	};

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
}