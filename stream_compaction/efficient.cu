#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernSetZero(int N, int* dev_data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			dev_data[index] = 0;
		}

		__global__ void kernEffcientUpSweep(int N, int offset, int* dev_data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			//if ((index + 1) % offset == 0) {
			//	dev_data[index] += dev_data[index - offset / 2];
			//}

			int targetIndex = (index + 1) * offset - 1;
			dev_data[targetIndex] += dev_data[targetIndex - offset / 2];
		}

		__global__ void kernSetRootZero(int N, int* dev_data) {
			dev_data[N - 1] = 0;
		}

		__global__ void kernEfficientDownSweep(int N, int offset, int* dev_data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			//if ((index + 1) % offset == 0) {
			//	int t = dev_data[index - offset / 2];
			//	dev_data[index - offset / 2] = dev_data[index];
			//	dev_data[index] += t;
			//}

			int targetIndex = (index + 1) * offset - 1;

			int t = dev_data[targetIndex - offset / 2];
			dev_data[targetIndex - offset / 2] = dev_data[targetIndex];
			dev_data[targetIndex] += t;
		}

		__global__ void kernSetCompactCount(int N, int* dev_count, int* bools, int* indices) {
			dev_count[0] = bools[N - 1] ? (indices[N - 1] + 1) : indices[N - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int dMax = ilog2ceil(n);
			int size = (int)powf(2.0f, (float)dMax);

			int* dev_data;

			cudaMalloc((void**)&dev_data, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaDeviceSynchronize();

			dim3 blockDim(blockSize);
			dim3 gridDim((size + blockSize - 1) / blockSize);

			kernSetZero << < gridDim, blockDim >> > (size, dev_data);
			checkCUDAError("kernSetZero failed!");

			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("efficient cudaMemcpy failed!");


            timer().startGpuTimer();
            // TODO

			// Step 1 : Up-sweep
			for (int d = 0; d <= dMax - 1; d++) {
				//kernEffcientUpSweep << <gridDim, blockDim >> > (size, (int)powf(2.0f, (float)d + 1.0f), dev_data);
				
				//only launch threads that acutally work
				int temp_size = (int)powf(2.0f, (float)(dMax - d - 1));
				kernEffcientUpSweep << <dim3((temp_size + blockSize - 1) / blockSize), blockDim >> > (temp_size, (int)powf(2.0f, (float)d + 1.0f), dev_data);

			}
			checkCUDAError("kernEffcientUpSweep failed!");

			// Step 2 : Down-sweep
			kernSetRootZero << < dim3(1), dim3(1) >> > (size, dev_data);
			checkCUDAError("kernSetRootZero failed!");

			for (int d = dMax - 1; d >= 0; d--) {
				//kernEfficientDownSweep << <gridDim, blockDim >> > (size, (int)powf(2.0f, (float)d + 1.0f), dev_data);
				
				//only launch threads that acutally work
				int temp_size = (int)powf(2.0f, (float)(dMax - d - 1));
				kernEfficientDownSweep << <dim3((temp_size + blockSize - 1) / blockSize), blockDim >> > (temp_size, (int)powf(2.0f, (float)d + 1.0f), dev_data);
			}
			checkCUDAError("kernEfficientDownSweep failed!");

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("efficient cudaMemcpy failed!");
			
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
			// compact Set-up
			int* dev_idata;
			int* dev_odata;
			int* bools;
			int* indices;
			int* dev_count;
			int count;

			dim3 blockDim(blockSize);
			dim3 gridDim((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&bools, n * sizeof(int));
			checkCUDAError("cudaMalloc bools failed!");
			cudaMalloc((void**)&dev_count, sizeof(int));
			checkCUDAError("cudaMalloc dev_count failed!");
			cudaDeviceSynchronize();

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("efficient compact cudaMemcpy failed!");



			// scan Set-up
			int dMax = ilog2ceil(n);
			int size = (int)powf(2.0f, (float)dMax);

			dim3 scan_gridDim((size + blockSize - 1) / blockSize);

			cudaMalloc((void**)&indices, size * sizeof(int));
			checkCUDAError("cudaMalloc indices failed!");
			cudaDeviceSynchronize();

			kernSetZero << < scan_gridDim, blockDim >> > (size, indices);
			checkCUDAError("kernSetZero failed!");


            timer().startGpuTimer();
            // TODO
			// Step 1 : compute bools array
			StreamCompaction::Common::kernMapToBoolean << <gridDim, blockDim >> > (n, bools, dev_idata);
			checkCUDAError("kernMapToBoolean failed!");

			cudaMemcpy(indices, bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy failed!");

			// Step 2 : exclusive scan indices
			// Up-sweep
			for (int d = 0; d <= dMax - 1; d++) {
				//kernEffcientUpSweep << <scan_gridDim, blockDim >> > (size, (int)powf(2.0f, (float)d + 1.0f), indices);
				
				//only launch threads that acutally work
				int temp_size = (int)powf(2.0f, (float)(dMax - d - 1));
				kernEffcientUpSweep << <dim3((temp_size + blockSize - 1) / blockSize), blockDim >> > (temp_size, (int)powf(2.0f, (float)d + 1.0f), indices);
			}
			checkCUDAError("kernEffcientUpSweep failed!");

			// Down-sweep
			kernSetRootZero << < dim3(1), dim3(1) >> > (size, indices);
			checkCUDAError("kernSetRootZero failed!");

			for (int d = dMax - 1; d >= 0; d--) {
				//kernEfficientDownSweep << <scan_gridDim, blockDim >> > (size, (int)powf(2.0f, (float)d + 1.0f), indices);
				
				//only launch threads that acutally work
				int temp_size = (int)powf(2.0f, (float)(dMax - d - 1));
				kernEfficientDownSweep << <dim3((temp_size + blockSize - 1) / blockSize), blockDim >> > (temp_size, (int)powf(2.0f, (float)d + 1.0f), indices);
			}
			checkCUDAError("kernEfficientDownSweep failed!");

			// Step 3 : Scatter
			StreamCompaction::Common::kernScatter << <gridDim, blockDim >> > (n, dev_odata, dev_idata, bools, indices);
			checkCUDAError("kernScatter failed!");

			kernSetCompactCount << <dim3(1), dim3(1) >> > (n, dev_count, bools, indices);
			checkCUDAError("kernSetCompactCount failed!");

            timer().endGpuTimer();


			cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaMemcpy(odata, dev_odata, sizeof(int) * count, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(bools);
			cudaFree(dev_count);
			cudaFree(indices);

            return count;
        }
    }
}
