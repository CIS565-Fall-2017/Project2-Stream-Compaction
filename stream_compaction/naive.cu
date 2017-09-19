#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kernNaiveScan(int n, int i, int* dev_idata, int* dev_odata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) 
			{
				return;
			}
			int p = 1 << (i - 1);
			if (index >= p)
			{
				dev_odata[index] = dev_idata[index - p] + dev_idata[index];
			}
			else
			{
				dev_odata[index] = dev_idata[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
			int *dev_idata, *dev_odata;
			int m = ilog2ceil(n);
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			for (int i = 1; i <= m; i++)
			{
				kernNaiveScan << <fullBlocksPerGrid, blockSize >> >(n, i, dev_idata, dev_odata);
				std::swap(dev_idata, dev_odata);
			}
			std::swap(dev_idata, dev_odata);
			timer().endGpuTimer();
			//inclusive->exclusive
			cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;
			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
