#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "RadixSort.h"

namespace RadixSort {
	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	__global__ void kernGen_b_e_array(int N, int idxBit, int* b_array, int* e_array, const int *dev_data) {
		// TODO
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= N) {
			return;
		}

		int temp_result = (dev_data[index] >> idxBit) & 1;
		b_array[index] = temp_result;
		e_array[index] = 1 - temp_result;

	}

	template<int BLOCK_SIZE>
	__global__ void kern_Gen_d_array_and_scatter(int N, const int totalFalses, const int* b_array, const int* f_array, int* dev_data)
	{
		//Allocate appropriate shared memory
		__shared__ int tile[BLOCK_SIZE];
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= N) {
			return;
		}

		tile[threadIdx.x] = dev_data[index];
		__syncthreads();

		int t_array_value = index - f_array[index] + totalFalses;

		int d_array_value = b_array[index] ? t_array_value : f_array[index];

		dev_data[d_array_value] = tile[threadIdx.x];
	}


	void sort(int n, int numOfBits, int *odata, const int *idata) {

		int* dev_data;
		int* b_array;
		int* e_array;
		int* f_array;

		int* host_f_array = new int[n];

		dim3 blockDim(blockSize);
		dim3 gridDim((n + blockSize - 1) / blockSize);

		cudaMalloc((void**)&dev_data, n * sizeof(int));
		checkCUDAError("cudaMalloc dev_data failed!");
		cudaMalloc((void**)&b_array, n * sizeof(int));
		checkCUDAError("cudaMalloc b_array failed!");
		cudaMalloc((void**)&e_array, n * sizeof(int));
		checkCUDAError("cudaMalloc e_array failed!");
		cudaMalloc((void**)&f_array, n * sizeof(int));
		checkCUDAError("cudaMalloc f_array failed!");
		cudaDeviceSynchronize();

		cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
		checkCUDAError("RadixSort cudaMemcpy failed!");

		timer().startGpuTimer();
		
		for (int k = 0; k <= numOfBits - 1; k++) {
			kernGen_b_e_array << <gridDim, blockDim >> > (n, k, b_array, e_array, dev_data);

			cudaMemcpy(host_f_array, e_array, sizeof(int) * n, cudaMemcpyDeviceToHost);

			int totalFalses = host_f_array[n - 1];

			// Get Exclusive scan result as a whole
			StreamCompaction::Efficient::scan(n, host_f_array, host_f_array);

			totalFalses += host_f_array[n - 1];

			cudaMemcpy(f_array, host_f_array, sizeof(int) * n, cudaMemcpyHostToDevice);

			// Since here we run exclusive scan as a whole,
			// and we don't want each tile to run StreamCompaction::Efficient::scan individually.
			// value in d_array here is actually index value in the whole data array, not just index in that tile
			// so, there is NO need to merge here
			kern_Gen_d_array_and_scatter<blockSize> << <gridDim, blockDim >> > (n, totalFalses, b_array, f_array, dev_data);
		}

		timer().endGpuTimer();

		cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy failed!");

		cudaFree(dev_data);
		cudaFree(b_array);
		cudaFree(e_array);
		cudaFree(f_array);

		delete[] host_f_array;
	}
}