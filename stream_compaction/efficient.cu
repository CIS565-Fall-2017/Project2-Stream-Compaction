#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

static int blockSize = 128;
static int blockNum;

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
		__global__ void cudaSweepUp(int n, int d, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);		
			int interval_length = 1 << (d + 1);
			if (index * interval_length >= n)
				return;
			int idx1 = index * interval_length + (1 << (d + 1)) - 1;
			int idx2 = index * interval_length + (1 << d) - 1;
			data[idx1] += data[idx2];
		}

		__global__ void cudaSweepDown(int n, int d, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int interval_length = 1 << (d + 1);
			// k from 0 to n-1
			if (index * interval_length >= n)
				return;

			int temp = data[index * interval_length + (1 << d) - 1];
			data[index * interval_length + (1 << d) - 1] = data[index * interval_length + (1 << (d + 1)) - 1];
			data[index * interval_length + (1 << (d + 1)) - 1] += temp;
		}

        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			if (n <= 0)
				return;
			int celllog = ilog2ceil(n);

			int pow2len = 1 << celllog;

			int *dev_data;
			cudaMalloc((void**)&dev_data, pow2len * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");

			cudaMemcpy(dev_data, idata, pow2len * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");

			//Up-Sweep
			for (int d = 0; d <= celllog - 1; d++) {
				int flag = (1 << (d + 1));
				blockNum = (pow2len / flag + blockSize) / blockSize;
				cudaSweepUp<<<blockNum, blockSize>>>(pow2len, d, dev_data);
			}
			
			//cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			//Down-Sweep
			cudaMemset(dev_data + pow2len - 1, 0, sizeof(int));
			checkCUDAError("cudaMemset failed!");

			for (int d = celllog - 1; d >= 0; d--) {
				int flag = (1 << (d + 1));
				blockNum = (pow2len / flag + blockSize) / blockSize;
				cudaSweepDown<<<blockNum, blockSize >>>(pow2len, d, dev_data);
			}

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			checkCUDAError("cudaMalloc dev_data[1] failed!");

			cudaFree(dev_data);
			checkCUDAError("cudaMalloc dev_data failed!");

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
            timer().endGpuTimer();
            return -1;
        }
    }
}
