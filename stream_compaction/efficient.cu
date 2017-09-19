#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#define BLOCK_SIZE 896
namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		__global__ void upSweep(const int n, const int step, int *data) {

			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) {
				return;
			}
			int rIndex = n - 1 - index;
			if (index - step >= 0 && (rIndex % (step * 2) == 0)) {
				data[index] = data[index] + data[index - step];
			}
			__syncthreads();
		}

		__global__ void downSweep(const int n, const int step, int *data) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) {
				return;
			}
			int rIndex = n - 1 - index;
			///Only certain index is working.
			if (index - step >= 0 && (rIndex % (step * 2) == 0) ) {
				auto tmp = data[index];
				data[index] += data[index - step];
				data[index - step] = tmp;
			}
			__syncthreads();
		}

		void scanOnGPU(const int n, int *dev_data) {
			dim3 blockCount = (n - 1) / BLOCK_SIZE + 1;
			int step;
			for (step = 1; step < n; step <<= 1) {
				upSweep << <blockCount, BLOCK_SIZE >> >(n, step, dev_data);
			}
			cudaMemset(&dev_data[n - 1], 0, sizeof(int));
			for (step >>= 1; step > 0; step >>= 1) {
				downSweep << <blockCount, BLOCK_SIZE >> >(n, step, dev_data);
			}
		}

		void scan(int n, int *odata, const int *idata) {
			// TODO
			int *dev_data;
			cudaMalloc((void**)&dev_data, sizeof(int) * n);
			cudaMemcpy((void*)dev_data, (const void*)idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			scanOnGPU(n, dev_data);
			cudaMemcpy((void*)odata, (const void*)dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_data);
		}

		/**
		* Performs stream compaction on idata, storing the result into odata.
		* All zeroes are discarded.
		*
		* @param n      The number of elements in idata.
		* @param odata  The array into which to store elements.
		* @param idata  The array of elements to compact.
		* @returns      The number of elements remaining after compaction.
		*/
		int compact(int n, int *odata, const int *idata) {
			// TODO
			int count = 0;
			int *dev_data;
			int *dev_dataCopy;
			int *dev_bool;
			int *dev_indices;
			for (int i = 0; i < n; ++i)
				count = count + (idata[i] != 0);

			// device memory allocation
			timer().startGpuTimer();

			cudaMalloc((void**)&dev_data, sizeof(int) * n);
			cudaMalloc((void**)&dev_dataCopy, sizeof(int) * n);
			cudaMalloc((void**)&dev_bool, sizeof(int) * n);
			cudaMalloc((void**)&dev_indices, sizeof(int) * n);
			// copy input data to device
			cudaMemcpy((void*)dev_data, (const void*)idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			dim3 blockCount = (n - 1) / BLOCK_SIZE + 1;
			Common::kernMapToBoolean << <blockCount, BLOCK_SIZE >> >(n, dev_bool, dev_data);
			cudaMemcpy((void*)dev_indices, (const void*)dev_bool, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			scanOnGPU(n, dev_indices);
			cudaMemcpy((void*)dev_dataCopy, (const void*)dev_data, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			Common::kernScatter << <blockCount, BLOCK_SIZE >> >(n, dev_data, dev_dataCopy, dev_bool, dev_indices);
			// copy result to host
			cudaMemcpy((void*)odata, (const void*)dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

			timer().endGpuTimer();
			// free memory on device
			cudaFree(dev_data);
			cudaFree(dev_dataCopy);
			cudaFree(dev_bool);
			cudaFree(dev_indices);

			return count;
		}
	}
}