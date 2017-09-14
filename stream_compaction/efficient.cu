#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweepIteration(int n, int d, int *idata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			k = k * (1 << (d + 1)) - 1;
			if (k > n || k < 0) { return; }

			int offset = 1 << d + 1;
			int old_val = idata[k];

			
			idata[k] = idata[k] + idata[k - (offset / 2)];

			//printf("d = %i, %i off %i %i oldval: %i val: %i\n", d, k, offset, offset / 2,old_val, idata[k]);
			
		}

		__global__ void kernDownSweepIteration(int n, int d, int *idata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			k = k * (1 << (d + 1)) - 1;
			if (k > n || k < 0) { return; }

			int offset = 1 << d + 1;
			int add = idata[k] + idata[k - (offset / 2)];
			int replace = idata[k];
			int old_val = idata[k];
			idata[k - (offset / 2)] = replace;
			idata[k] = add;
			printf("d = %i, %i off %i %i oldval: %i val: %i\n", d, k, offset, offset / 2, old_val, idata[k]);


		}

		__global__ void kernSetZero(int n, int *idata) {
			idata[n] = 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			dim3 threadsPerBlock(128);
			int blockSize = 128;
			

			
			int arr_size = n;
			int log = ilog2ceil(arr_size);
			if ((n & (n - 1)) != 0) {
				arr_size = 1 << log;
			}
			printf("n: %i arr_size: %i \n", n, arr_size);
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, arr_size * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			int* dev_shiftdata;
			cudaMalloc((void**)&dev_shiftdata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_shiftdata failed!");

			cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			//UPSWEEP
			log = ilog2ceil(arr_size);
			for (int d = 0; d < log; d++) {
				int off_n = arr_size / (1 << (d + 1));
				printf("num %i \n", off_n);
				dim3 fullBlocksPerGrid((off_n + blockSize - 1) / blockSize);
				kernUpSweepIteration << <fullBlocksPerGrid, blockSize >> > (arr_size, d, dev_odata);
			}
			//DOWNSWEEP
			kernSetZero <<<1, 1 >>> (arr_size - 1, dev_odata);
			for (int d = log - 1; d >= 0; d--) {
				int off_n = arr_size / (1 << (d + 1));
				dim3 fullBlocksPerGrid((off_n + blockSize - 1) / blockSize);
				kernDownSweepIteration << <fullBlocksPerGrid, blockSize >> > (arr_size, d, dev_odata);
			}
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy failed!");
            timer().endGpuTimer();
			cudaFree(dev_odata);
			cudaFree(dev_shiftdata);
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
