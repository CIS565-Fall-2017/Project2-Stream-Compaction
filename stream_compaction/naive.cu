#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <math.h> 

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernNaiveScanByLevel(int N, int* scan_out, int* scan_in, int offset)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index >= N) 
			{
				return;
			}

			scan_out[index] = index >= offset ? scan_in[index] + scan_in[index - offset] : scan_in[index];
		}

		__global__ void kernConvertToExclusive(int N, int* scan_out, int* scan_in)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index >= N)
			{
				return;
			}

			scan_out[index] = index == 0? 0 : scan_in[index - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
		{
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize); //To ensure if N is not an exact multiple of blocksize, 
			
			const int numBytes = n * sizeof(int);

			int* dev_scan_out;
			int* dev_scan_in;

			cudaMalloc((void**)&dev_scan_out, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_scan_out failed!");
			cudaMalloc((void**)&dev_scan_in, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_scan_out failed!");

			cudaMemcpy(dev_scan_out, idata, numBytes, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_scan_A failed!");
			cudaMemcpy(dev_scan_in, idata, numBytes, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_scan_A failed!");

			timer().startGpuTimer();

				int log_n = ilog2ceil(n);
				for (int i = 1; i <= log_n; i++)
				{
					int offset = 1 << i - 1;
					kernNaiveScanByLevel <<<fullBlocksPerGrid, blockSize>>> (n, dev_scan_out, dev_scan_in, offset);
					std::swap(dev_scan_out, dev_scan_in);
				}

				kernConvertToExclusive <<<fullBlocksPerGrid, blockSize>>> (n, dev_scan_out, dev_scan_in);

            timer().endGpuTimer();

			cudaThreadSynchronize();

			cudaMemcpy(odata, dev_scan_out, numBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_odata to odata failed!");

			cudaFree(dev_scan_out);
			cudaFree(dev_scan_in);
			checkCUDAErrorFn("cudaFree failed!");
        }
    }
}
