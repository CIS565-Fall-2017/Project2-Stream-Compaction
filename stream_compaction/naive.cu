#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

int *devi, *devo;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		__global__ void kernelNaive(int n, int delta, const int *idata, int *odata) 
		{
			int index = (blockIdx.x *blockDim.x) + threadIdx.x;
			if (index >= n) 
			{
				return;
			}
			if (index - delta < 0) 
			{
				odata[index] = idata[index];
			}
			else 
			{
				odata[index] = idata[index - delta] + idata[index];
			}
		}



        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
			int depth = ilog2ceil(n);

			cudaMalloc((void**)&devi, n * sizeof(int));
			cudaMalloc((void**)&devo, n * sizeof(int));

			checkCUDAError("cudaMalloc error");

			cudaMemcpy(devi, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int blockNum = (n + blockSize - 1) / blockSize;
			int delta;

			timer().startGpuTimer();
			for (int i = 1; i <= depth; i++)
			{
				delta = (1 << (i - 1));
				kernelNaive << < blockNum, blockSize >> >(n, delta, devi, devo);
				std::swap(devi, devo);
			}

			timer().endGpuTimer();

			std::swap(devi, devo);

			cudaMemcpy(odata + 1, devo, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;

			cudaFree(devi);
			cudaFree(devo);            
        }
    }
}
