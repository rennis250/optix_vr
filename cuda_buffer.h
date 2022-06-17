#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "exceptions.h"

#include <iostream>

namespace rob {
	template <typename T>
	class CudaBuffer {
	public:
		CudaBuffer(int32_t width, int32_t height);
		CudaBuffer(const size_t size);
		CudaBuffer(const std::vector<T>& v);
		~CudaBuffer();

		void upload();
		void download();

		const unsigned int get_size() { return m_h_data.size(); };
		const size_t get_cuda_size() { return m_cuda_size; };

		std::vector<T> get_host_data() { return m_h_data; };
		T* get_device_ptr() { return m_d_data; };

	private:
		std::vector<T> m_h_data;
		T* m_d_data = nullptr;
		size_t m_cuda_size = 0;
	};

	template <typename T>
	CudaBuffer<T>::CudaBuffer(int32_t width, int32_t height) {
		m_cuda_size = sizeof(T) * (size_t)width * (size_t)height;
		m_h_data.resize((size_t)width * (size_t)height);
		try {
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_data), m_cuda_size));
		}
		catch (std::exception& e)
		{
			std::cerr << "CudaBuffer error: " << e.what() << std::endl;
		}
	}

	template <typename T>
	CudaBuffer<T>::CudaBuffer(const size_t size) {
		m_cuda_size = sizeof(T) * size;
		m_h_data.resize(size);
		try {
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_data), m_cuda_size));
		}
		catch (std::exception& e)
		{
			std::cerr << "CudaBuffer error: " << e.what() << std::endl;
		}
	}

	template <typename T>
	CudaBuffer<T>::CudaBuffer(const std::vector<T>& v) {
		m_cuda_size = sizeof(T) * v.size();
		m_h_data = v;
		try {
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_data), m_cuda_size));
		}
		catch (std::exception& e)
		{
			std::cerr << "CudaBuffer error: " << e.what() << std::endl;
		}
	}

	template <typename T>
	void CudaBuffer<T>::upload() {
		try {
			CUDA_CHECK(
				cudaMemcpy(reinterpret_cast<void*>(m_d_data),
					m_h_data.data(),
					m_cuda_size,
					cudaMemcpyHostToDevice)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "CudaBuffer error: " << e.what() << std::endl;
		}

		return;
	}
	
	template <typename T>
	void CudaBuffer<T>::download() {
		try {
			CUDA_CHECK(
				cudaMemcpy(
					reinterpret_cast<void*>(m_h_data.data()),
					m_d_data,
					m_cuda_size,
					cudaMemcpyDeviceToHost
				)
			);
		}
		catch (std::exception& e)
		{
			std::cerr << "CudaBuffer error: " << e.what() << std::endl;
		}

		return;
	}

	template <typename T>
	CudaBuffer<T>::~CudaBuffer() {
		try
		{
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_data)));
		}
		catch (std::exception& e)
		{
			std::cerr << "CudaBuffer error: " << e.what() << std::endl;
		}
	}
}