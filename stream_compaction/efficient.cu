#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int d, int d1, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int k = d1 * index;
			idata[k + d1 - 1] += idata[k + d - 1];
		}

		__global__ void kernDownSweep(int d, int d1, int*idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int k = d1*index;
			int t = idata[k + d - 1];
			idata[k + d - 1] = idata[k + d1 - 1];
			idata[k + d1 - 1] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

			int *dev_iData;

			// smallest power of 2 >= n
			int pow2 = pow(2, ilog2ceil(n));
			int levels = ilog2ceil(n);
			cudaMalloc((void**)&dev_iData, (pow2 + 1) * sizeof(int));
			cudaMemcpy(dev_iData, idata, sizeof(int)*n, cudaMemcpyHostToDevice);


			for (int i = 0; i < levels; i++) {
				int d = pow(2, i);
				int d1 = pow(2, i + 1);
				
				int blocknum = ceil(pow2/ d1);
				dim3 fullBlocks((blocknum + blockSize) / blockSize);

				kernUpSweep << <fullBlocks, blockSize>> > (d, d1, dev_iData);
				cudaThreadSynchronize();
			}

			int a = 0;
			cudaMemcpy(&dev_iData[pow2 - 1], &a, sizeof(int), cudaMemcpyHostToDevice);
			for (int i = levels - 1; i >= 0; i--) {
				int d = pow(2, i);
				int d1 = pow(2, i + 1);

				int blocknum = ceil(pow2 / d1);
				dim3 fullBlocks((blocknum + blockSize) / blockSize);

				kernDownSweep << <fullBlocks, blockSize>> > (d, d1, dev_iData);
				cudaThreadSynchronize();
			}


			cudaMemcpy(odata, dev_iData, sizeof(int)*(n), cudaMemcpyDeviceToHost);
			cudaFree(dev_iData);

            timer().endGpuTimer();
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
            // TODO
			int *dev_bools;
			int *dev_indices;
			int *dev_odata;
			int *dev_idata;

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_idata, n * sizeof(int));

			cudaMemcpy(dev_idata, idata, sizeof(int)*n, cudaMemcpyHostToDevice);

			dim3 otherName((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <otherName, blockSize >> > (n, dev_bools, dev_idata);

			int *indices = new int[n];
			int *bools = new int[n];

			cudaMemcpy(bools, dev_bools, sizeof(int)*n, cudaMemcpyDeviceToHost);

			timer().endGpuTimer();
			scan(n, indices, bools);
			timer().startGpuTimer();

			cudaMemcpy(dev_indices, indices, sizeof(int)*n, cudaMemcpyHostToDevice);


			StreamCompaction::Common::kernScatter << <otherName, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

			int count;
			cudaMemcpy(&count, &dev_indices[n-1], sizeof(int), cudaMemcpyDeviceToHost);

			int lastBool;
			cudaMemcpy(&lastBool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
			

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);


			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			delete[] bools;
			delete[] indices;
			timer().endGpuTimer();
            return count + lastBool;
        }
    }
}
