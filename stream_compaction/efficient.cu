#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int n, int offset, int *odata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;

			if (idx >= n) return;

			if (n == 1)
			{
				odata[offset - 1] = 0;
				return;
			}

			int cur = (idx + 1) * offset - 1;

			int prev = cur - (offset / 2);

			odata[cur] += odata[prev];
		}

		__global__ void kernDownSweep(int n, int offset, int *odata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;

			if (idx >= n) return;

			int cur = (idx + 1) * offset - 1;

			int prev = cur - (offset / 2);

			int temp = odata[prev];
			odata[prev] = odata[cur];
			odata[cur] += temp;
		}

		int getPadded(int n) {
			int countOfOnes = 0;
			int ret = 1;
			while (n != 1)
			{
				if (n & 1 == 1)
				{
					++countOfOnes;
				}
				n >>= 1;
				ret <<= 1;
			}
			if (countOfOnes == 0) return ret;
			else return ret << 1;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *dev_odata;
			dev_odata = nullptr;

			int numToCompute = getPadded(n);

			cudaMalloc(&dev_odata, numToCompute * sizeof(int));
			cudaMemset(dev_odata, 0, numToCompute);
			cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int depth = ilog2ceil(n);
			int blockSize = 256;
			int offset = 1;
			dim3 threadPerBlock(blockSize);

			timer().startGpuTimer();

			for (int i = 0; i < depth; ++i)
			{
				numToCompute /= 2;
				offset *= 2;
				int blocksPerGrid = (numToCompute + blockSize - 1) / blockSize;
				kernUpSweep << <blocksPerGrid, blockSize >> > (numToCompute, offset, dev_odata);

			}

			numToCompute = 1;
			for (int i = 0; i < depth; ++i)
				{
				int blocksPerGrid = (numToCompute + blockSize - 1) / blockSize;
				kernDownSweep << <blocksPerGrid, blockSize >> > (numToCompute, offset, dev_odata);
				numToCompute *= 2;
				offset /= 2;
			}

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaDeviceSynchronize();
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
