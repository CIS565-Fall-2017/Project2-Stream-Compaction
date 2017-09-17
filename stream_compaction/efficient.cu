#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernEfficientUpsweep(int pow2plus1, int pow2, int N, int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < N) {
				if (index % pow2plus1==0) {
					idata[index + pow2plus1 - 1] += idata[index + pow2 - 1];
				}
			}
		}

		__global__ void kernEfficientDownsweep(int pow2plus1, int pow2, int N, int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < N) {
				if (index % pow2plus1 == 0) {
					int t = idata[index + pow2 - 1];
					idata[index + pow2 - 1] = idata[index + pow2plus1 - 1];
					idata[index + pow2plus1 - 1] += t;
				}
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int *dev_tempin;
			cudaMalloc((void**)&dev_tempin, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_tempin failed!");
			cudaMemcpy(dev_tempin, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("cuda Memcpy from idata to dev_tempin failed!");

			timer().startGpuTimer();
			for (int iteration = 0; iteration <= ilog2ceil(n)-1; iteration++) {
					kernEfficientUpsweep << <fullBlocksPerGrid, blockSize >> > (pow(2, iteration + 1),
						pow(2, iteration), n, dev_tempin);
			}

			cudaMemcpy(odata, dev_tempin, n * sizeof(int), cudaMemcpyDeviceToHost);
			odata[n - 1] = 0;
			cudaMemcpy(dev_tempin, odata, n * sizeof(int), cudaMemcpyHostToDevice);

			for (int iteration = ilog2ceil(n) - 1; iteration >= 0; iteration--) {
				kernEfficientDownsweep << <fullBlocksPerGrid, blockSize >> > (pow(2, iteration + 1),
					pow(2, iteration), n, dev_tempin);
			}
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_tempin, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("cuda Memcpy from dev_tempin to odata failed!");
			cudaFree(dev_tempin);
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
            timer().endGpuTimer();
            return -1;
        }
    }
}
