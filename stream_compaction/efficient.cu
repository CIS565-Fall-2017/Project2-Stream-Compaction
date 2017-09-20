#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int n, int d, int d1, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index > (n/d1)) {
				return;
			}
			int k = d1 * index;
			idata[k + d1 - 1] += idata[k + d - 1];
		}

		__global__ void kernDownSweep(int n, int d, int d1, int*idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index > (n / d1)) {
				return;
			}
			int k = d1*index;
			int t = idata[k + d - 1];
			idata[k + d - 1] = idata[k + d1 - 1];
			idata[k + d1 - 1] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO

			int *dev_iData;

			// smallest power of 2 >= n
			int pow2 = pow(2, ilog2ceil(n));
			int levels = ilog2ceil(n);
			cudaMalloc((void**)&dev_iData, (pow2 + 1) * sizeof(int));
			checkCUDAError("cudaMalloc dev_iData failed");

			cudaMemcpy(dev_iData, idata, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_iData failed");

			timer().startGpuTimer();

			for (int i = 0; i < levels; i++) {
				int d = pow(2, i);
				int d1 = pow(2, i + 1);
				
				int blocknum = ceil(pow2/ d1);
				dim3 fullBlocks((blocknum + blockSize - 1) / blockSize);

				kernUpSweep << <fullBlocks, blockSize>> > (n, d, d1, dev_iData);
				cudaThreadSynchronize();
			}

			int a = 0;
			cudaMemcpy(&dev_iData[pow2 - 1], &a, sizeof(int), cudaMemcpyHostToDevice);
			for (int i = levels - 1; i >= 0; i--) {
				int d = pow(2, i);
				int d1 = pow(2, i + 1);

				int blocknum = ceil(pow2 / d1);
				dim3 fullBlocks((blocknum + blockSize - 1) / blockSize);

				kernDownSweep << <fullBlocks, blockSize>> > (n, d, d1, dev_iData);
				cudaThreadSynchronize();
			}

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_iData, sizeof(int)*(n), cudaMemcpyDeviceToHost);
			cudaFree(dev_iData);

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
			int *dev_bools;
			int *dev_indices;
			int *dev_odata;
			int *dev_idata;

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed");

			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed");
			
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed");
			
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed");

			cudaMemcpy(dev_idata, idata, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed");


			timer().startGpuTimer();

			dim3 otherName((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <otherName, blockSize >> > (n, dev_bools, dev_idata);

			int *indices = new int[n];
			int *bools = new int[n];

			cudaMemcpy(bools, dev_bools, sizeof(int)*n, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_bools failed");

			timer().endGpuTimer();
			scan(n, indices, bools);
			timer().startGpuTimer();

			cudaMemcpy(dev_indices, indices, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_indices failed");


			StreamCompaction::Common::kernScatter << <otherName, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
			timer().endGpuTimer();

			int count;
			cudaMemcpy(&count, &dev_indices[n-1], sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_indices failed");

			int lastBool;
			cudaMemcpy(&lastBool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_bools failed");

			
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_bools failed");

			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			delete[] bools;
			delete[] indices;
            return count + lastBool;
        }


    }
}
