#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

		#define blockSize 128

		__global__ void kernScanShift(int N, int offset, int * odata, int * idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index >= N)
				return;

			int res = idata[index];

			if (index >= offset)
				res += idata[index - offset];

			index += 1;

			if (index < N)
				odata[index] = res;
		}

		__global__ void kernScan(int N, int offset, int * odata, int * idata) 
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index >= N)
				return;

			int res = idata[index];
			
			if (index >= offset)
				res += idata[index - offset];

			odata[index] = res;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int * dev_ping;
			int * dev_pong;

			cudaMalloc((void**)&dev_ping, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_ping failed!");

			cudaMalloc((void**)&dev_pong, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_pong failed!");

			// Only ping is memcpyed
			cudaMemcpy(dev_ping, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy dev_ping failed!");
			
			dim3 blocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
			int passes = ilog2ceil(n);

			for (int d = 0; d < passes; ++d)
			{
				int offset = pow(2, d); // (d-1)+1

				if (d == passes - 1)
				{
					kernScanShift << <blocks, blockSize >> > (n, offset, dev_pong, dev_ping);
					checkCUDAErrorFn("kernScanShift failed!");
				}
				else
				{
					kernScan << <blocks, blockSize >> > (n, offset, dev_pong, dev_ping);
					checkCUDAErrorFn("kernScan failed!");
				}

				std::swap(dev_pong, dev_ping);
			}
            timer().endGpuTimer();

			cudaMemcpy(odata, dev_ping, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy dev_ping failed!");

			// I decided to do this so that we don't thrash the gpu memory access by jumping
			// For very large arrays this should be more efficient than doing it on gpu
			odata[0] = 0;

			cudaFree(dev_pong);
			cudaFree(dev_ping);
        }
    }
}
