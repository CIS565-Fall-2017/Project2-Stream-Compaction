#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
		__global__ void kernScanNaive(int N, int stride, int *odata, const int *idata)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx >= N) return;

			if (idx >= stride)
			{
				odata[idx] = idata[idx - stride] + idata[idx];
			}
			else
			{
				odata[idx] = idata[idx];
			}
		}

		__global__ void kernInclusiveToExclusive(int N, int *odata, const int *idata)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx >= N) return;

			odata[idx] = idx == 0 ? 0 : idata[idx - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			//Dimensions
			int blockSize = 256;
			int depth = ilog2ceil(n);
			int blocksPerGrid = (n + blockSize - 1) / blockSize;

			//Memory allocation
			int *dev_idata, *dev_odata;

			dev_idata = nullptr;
			dev_odata = nullptr;
			cudaMalloc(&dev_idata, n * sizeof(int));
			cudaMalloc(&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			int stride = 1;
			for (int d = 0; d < depth; ++d) {
				kernScanNaive << <blocksPerGrid, blockSize >> >(n, stride, dev_odata, dev_idata);
				int *temp = dev_odata;
				dev_odata = dev_idata;
				dev_idata = temp;
				stride *= 2;
			}

			kernInclusiveToExclusive << < blocksPerGrid, blockSize >> > (n, dev_odata, dev_idata);

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(&dev_idata);
			cudaFree(&dev_odata);
			cudaDeviceSynchronize();
        }
    }
}
