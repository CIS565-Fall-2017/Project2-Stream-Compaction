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

		__global__ void kernUpwardSweep(int n, int i, int *dev_data) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}

			int power = 1 << i;
			int powerPlusOne = 1 << (i + 1);
			if (index % powerPlusOne == powerPlusOne - 1) {
				dev_data[index] += dev_data[index - power];
			}

			if (index == n - 1) {
				dev_data[index] = 0;
			}
		}

		__global__ void kernDownwardSweep(int n, int arrayLength, int i, int *dev_data) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= arrayLength) {
				return;
			}
			
			int power = 1 << i;
			int powerMinusOne = 1 << (i - 1);
			if ((index + 1) % power == 0) {
				int temp = dev_data[index - powerMinusOne];
				dev_data[index - powerMinusOne] = dev_data[index];
				dev_data[index] += temp;
			}
		}

		void scanImplementation(int n, int logn, int arrayLength, int *dev_data) {
			dim3 blocksPerGrid((arrayLength + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			for (int i = 0; i < logn; i++) {
				kernUpwardSweep<<<blocksPerGrid, threadsPerBlock>>>(n, i, dev_data);
				checkCUDAError("kernUpwardSweep failed", __LINE__);
			}

			for (int i = logn; i > 0; i--) {
				kernDownwardSweep<<<blocksPerGrid, threadsPerBlock>>>(n, arrayLength, i, dev_data);
				checkCUDAError("kernDownwardSweep failed", __LINE__);
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int logn = ilog2ceil(n);
			int arrayLength = 1 << logn;
			int size = arrayLength * sizeof(int);

			// Allocate new idata with zeros
			int *idataFull = (int*)malloc(size);
			for (int i = 0; i < n; i++) {
				idataFull[i] = idata[i];
			}
			for (int i = n; i < arrayLength; i++) {
				idataFull[i] = 0;
			}

			// Allocate buffers
			int *dev_data;
			cudaMalloc((void**)&dev_data, size);
			checkCUDAError("cudaMalloc dev_data failed", __LINE__);

			// Copy input to device
			cudaMemcpy(dev_data, idataFull, size, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy input failed", __LINE__);

			// Scan
            timer().startGpuTimer();
			scanImplementation(n, logn, arrayLength, dev_data);
			timer().endGpuTimer();

			// Copy result from device
			cudaMemcpy(idataFull, dev_data, size, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy result failed", __LINE__);

			// Move results back to odata size
			for (int i = 0; i < n; i++) {
				odata[i] = idataFull[i];
			}

			// Free buffers and array
			free(idataFull);
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
            timer().startGpuTimer();

			int logn = ilog2ceil(n);
			int arrayLength = 1 << logn;
			int size = arrayLength * sizeof(int);
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((arrayLength + blockSize - 1) / blockSize);

			// Allocate device memory
			int *dev_bools, *dev_idata, *dev_indices, *dev_odata;

			cudaMalloc((void**)&dev_bools, size);
			checkCUDAError("cudaMalloc dev_bools failed", __LINE__);

			cudaMalloc((void**)&dev_idata, size);
			checkCUDAError("cudaMalloc dev_idata failed", __LINE__);

			cudaMalloc((void**)&dev_indices, size);
			checkCUDAError("cudaMalloc dev_indices failed", __LINE__);

			cudaMalloc((void**)&dev_odata, size);
			checkCUDAError("cudaMalloc dev_odata failed", __LINE__);
			
			// Allocate new idata with zeros
			int *idataFull = (int*)malloc(size);
			for (int i = 0; i < n; i++) {
				idataFull[i] = idata[i];
			}
			for (int i = n; i < arrayLength; i++) {
				idataFull[i] = 0;
			}

			// Copy input to device memory
			cudaMemcpy(dev_idata, idataFull, size, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed", __LINE__);

            // Create bools array
			StreamCompaction::Common::kernMapToBoolean<<<blocksPerGrid, threadsPerBlock>>>(arrayLength, dev_bools, dev_idata);
			checkCUDAError("kernMapToBoolean failed", __LINE__);

			// Scan bools array
			cudaMemcpy(dev_indices, dev_bools, size, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy dev_indices failed", __LINE__);

			scanImplementation(n, logn, arrayLength, dev_indices);

			// Scatter
			StreamCompaction::Common::kernScatter<<<blocksPerGrid, threadsPerBlock>>>(arrayLength, dev_odata, dev_idata, dev_bools, dev_indices);
			checkCUDAError("kernScatter failed", __LINE__);

			// Copy output from device memory
			cudaMemcpy(idataFull, dev_odata, size, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata failed", __LINE__);

			// Bring idataFull to odata
			int count = 0;
			for (int i = 0; i < n; i++) {
				int data = idataFull[i];
				if (data == 0) {
					break;
				}
				odata[i] = data;
				count++;
			}

			// Free memory
			free(idataFull);
			cudaFree(dev_idata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_odata);

            timer().endGpuTimer();
            return count;
        }
    }
}
