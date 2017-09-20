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
			cudaMemset(dev_odata, 0, numToCompute * sizeof(int));
			cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int depth = ilog2ceil(n);
			int blockSize = 1024;
			int offset = 1;

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
        }

		//__global__ void kernScanEachBlock(int n, int *a) {

		//}

		//void scanUsingSharedMem(int n, int *odata, const int *idata) {
		//	int numPadded = getPadded(n);

		//	int *dev_idata, *dev_odata;

		//	dev_idata = nullptr;
		//	cudaMalloc(&dev_idata, numPadded * sizeof(int));
		//	cudaMemset(dev_idata, 0, numPadded * sizeof(int));
		//	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

		//	dev_odata = nullptr;
		//	cudaMalloc(&dev_odata, numPadded * sizeof(int));

		//	int blockSize = 1024;
		//	int numOfBlocks = (numPadded + blockSize - 1) / blockSize;

		//	kernScanEachBlock << <numOfBlocks, blockSize >> > (blockSize, dev_idata, dev_odata);

		//	cudaFree(dev_idata);
		//}


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
			int numToCompute = getPadded(n);

			int *dev_odata, *dev_idata, *dev_bools, *dev_indices;
			dev_odata = nullptr;
			dev_idata = nullptr;
			dev_bools = nullptr;
			dev_indices = nullptr;

			cudaMalloc(&dev_bools, n * sizeof(int));
			cudaMalloc(&dev_idata, n * sizeof(int));
			cudaMalloc(&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(dev_bools, 0, n * sizeof(int));

			cudaMalloc(&dev_indices, numToCompute * sizeof(int));
			cudaMemset(dev_indices, 0, numToCompute * sizeof(int));

			int depth = ilog2ceil(n);
			int offset = 1;
			int blockSize = 1024;
			int blocksPerGrid = (n + blockSize - 1) / blockSize;

			timer().startGpuTimer();
			Common::kernMapToBoolean << <blocksPerGrid, blockSize >> > (numToCompute, dev_bools, dev_idata);
			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

			for (int i = 0; i < depth; ++i)
			{
				numToCompute /= 2;
				offset *= 2;
				blocksPerGrid = (numToCompute + blockSize - 1) / blockSize;
				kernUpSweep << <blocksPerGrid, blockSize >> > (numToCompute, offset, dev_indices);

			}

			numToCompute = 1;
			for (int i = 0; i < depth; ++i)
			{
				blocksPerGrid = (numToCompute + blockSize - 1) / blockSize;
				kernDownSweep << <blocksPerGrid, blockSize >> > (numToCompute, offset, dev_indices);
				numToCompute *= 2;
				offset /= 2;
			}

			int ret;
			cudaMemcpy(&ret, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			ret += idata[n - 1] == 0 ? 0 : 1;


			blocksPerGrid = (n + blockSize - 1) / blockSize;
			Common::kernScatter << <blocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, ret * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
            return ret;
        }
    }
}
