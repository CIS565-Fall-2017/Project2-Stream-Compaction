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

			if (idx % offset == 0)
			{
				odata[idx + offset - 1] += odata[idx + (offset >> 1) - 1];
			}
		}

		__global__ void kernDownSweep(int n, int offset, int *odata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;

			if (idx >= n) return;

			if (idx == n - 1) {
				odata[idx] = 0;
				return;
			}

			if (idx % offset == 0)
			{
				int temp = odata[idx + (offset >> 1) - 1];
				odata[idx + (offset >> 1) - 1] = odata[idx + offset - 1];
				odata[idx + offset - 1] += temp;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *dev_odata, *dev_idata;
			dev_odata = nullptr;
			dev_idata = nullptr;

			//cudaMalloc(&dev_idata, n * sizeof(int));
			cudaMalloc(&dev_odata, n * sizeof(int));
			//cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_odata, odata, n * sizeof(int), cudaMemcpyHostToDevice);

			int depth = ilog2ceil(n);
			int blockSize = 128;
			dim3 threadPerBlock(blockSize);
			dim3 blocksPerGrid((blockSize + n - 1) / blockSize);

			timer().startGpuTimer();

			for (int i = 0; i < depth; ++i)
			{
				kernUpSweep << <blocksPerGrid, threadPerBlock >> > (n, 1 << (i + 1), dev_odata);
			}

			for (int i = depth - 1; i >= 0; --i)
			{
				kernDownSweep << <blocksPerGrid, threadPerBlock >> > (n, 1 << (i + 1), dev_odata);
			}

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
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
