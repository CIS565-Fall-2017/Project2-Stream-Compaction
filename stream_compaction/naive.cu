#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		
        // TODO: __global__
		__global__ void naivescan(int n, int k, int* idev, int* odev)
		{
			auto index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {return;}
	 
			if (index >= k)
				odev[index] = idev[index] + idev[index - k];
			else
				odev[index] = idev[index];
		}


		__global__ void inc2exc(int n, int* idev, int* odev)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) { return; }

			if (index > 0)
				odev[index] = idev[index - 1];
			else
				odev[index] = 0;

		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int* idev;
			int* odev;
			cudaMalloc((void**)&idev, n * sizeof(int));
			checkCUDAError("Malloc for input device failed\n");

			cudaMalloc((void**)&odev, n * sizeof(int));
			checkCUDAError("Malloc for input device failed\n");

			cudaMemcpy(idev, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy for odataSwap failed");

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            // TODO
			int k = 1;
			for (int k = 1; k < n; k<<=1)
			{
				naivescan <<< fullBlocksPerGrid, blockSize >>> (n, k, idev, odev);
				std::swap(idev, odev);
			}

			inc2exc <<< fullBlocksPerGrid, blockSize >>> (n, idev, odev);
			
            timer().endGpuTimer();

			cudaMemcpy(odata, odev, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy for odataSwap failed");

			cudaFree(odev);
			cudaFree(idev);
        }
    }
}
