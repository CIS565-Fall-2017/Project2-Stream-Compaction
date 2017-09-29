#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient_Shared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernScanShared(int n, int *odata, int *sumOfSums)
		{
			extern __shared__ int temp[];
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int offset = 1;

			// load shared memory
			temp[2 * index] = odata[2 * index];
			temp[2 * index + 1] = odata[2 * index + 1];

			// upsweep
			for (int d = n >> 1; d > 0; d >>= 1) {
				__syncthreads();
				if (index < d) {
					int ai = offset * (2 * index + 1) - 1;
					int bi = offset * (2 * index + 2) - 1;

					temp[bi] += temp[ai];
				}
				offset *= 2;
			}

			if (index == 0) {
				sumOfSums[blockIdx.x] = temp[n - 1];
				temp[n - 1] = 0;
			}

			// downsweep
			for (int d = 1; d < n; d *= 2) {
				offset >>= 1;
				__syncthreads();
				if (index < d) {
					int ai = offset * (2 * index + 1) - 1;
					int bi = offset * (2 * index + 2) - 1;

					int t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			
			__syncthreads();

			odata[2 * index] = temp[2 * index];
			odata[2 * index + 1] = temp[2 * index + 1];
		}

		__global__ void kernAddSums(int n, int *odata, int *sumOfSums)
		{
			__shared__ int sum;
			if (threadIdx.x == 0) sum = sumOfSums[blockIdx.x];
			__syncthreads();
			
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			odata[index] += sum;
		}

		void scan_implementation(int n, int *dev_out, int *sumOfSums) 
		{
			int shMemBytes = sizeof(int) * blockSize;
			int numBlocks = (n + blockSize - 1) / blockSize;

			kernScanShared << <numBlocks, blockSize, shMemBytes >> > (numBlocks * blockSize, dev_out, sumOfSums);
			checkCUDAError("kernScanShared 1 failed!");
			
			if (n > blockSize) {
				kernScanShared << <numBlocks, blockSize, shMemBytes >> > (numBlocks, sumOfSums, sumOfSums);
				checkCUDAError("kernScanShared 2 failed!");

				kernAddSums << <numBlocks, blockSize >> > (pow2n, dev_out, sumOfSums);
				checkCUDAError("kernAddSums failed!");
			}
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata)
		{
			int *dev_out;
			int *sumOfSums;
			int pow2n = 1 << ilog2ceil(n);

			cudaMalloc((void**)&dev_out, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			cudaMalloc((void**)&sumOfSums, pow2n / blockSize * sizeof(int));
			checkCUDAError("cudaMalloc sumOfSums failed!");

			cudaMemset(dev_out, 0, pow2n * sizeof(int));
			checkCUDAError("cudaMemset dev_out failed!");

			cudaMemcpy(dev_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_out failed!");

			timer().startGpuTimer();
			scan_implementation(pow2n, dev_out, sumOfSums);
			timer().endGpuTimer();
			checkCUDAError("scan_implementation failed!");

			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			cudaFree(dev_out);
			cudaFree(sumOfSums);
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
        int compact(int n, int *odata, const int *idata) 
		{
			// TODO
			int *dbools;
			int *indices;
			int *dev_in;
			int *dev_out;
			int *sumOfSums;

			int pow2n = 1 << ilog2ceil(n);

			cudaMalloc((void**)&dbools, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc dbools failed!");

			cudaMalloc((void**)&indices, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc indices failed!");

			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in failed!");

			timer().startGpuTimer();

			dim3 blocksPerGrid1((pow2n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid1, blockSize >> > (pow2n, n, indices, dev_in);
			checkCUDAError("kernMapToBoolean failed!");

			cudaMemcpy(dbools, indices, pow2n * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpyDeviceToDevice failed!");

			int *num = (int *)malloc(sizeof(int));
			cudaMemcpy(num, dbools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			int ret = *num;

			cudaMalloc((void**)&sumOfSums, pow2n / blockSize * sizeof(int));
			checkCUDAError("cudaMalloc sumOfSums failed!");
		
			scan_implementation(pow2n, indices, sumOfSums); // requires power of 2

			cudaMemcpy(num, indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");
			ret += *num;
			free(num);

			cudaMalloc((void**)&dev_out, ret * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernScatter << <blocksPerGrid, blockSize >> > (n, dev_out, dev_in, dbools, indices);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_out, ret * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			cudaFree(dbools);
			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(indices);
			cudaFree(sumOfSums);
            
            return ret;
        }
    }
}
