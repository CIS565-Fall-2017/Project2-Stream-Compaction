#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		__global__ void kernScan(int n, int d, int *odata, int *idata) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= n) {
				return;
			}
			if (k >= d) {
				int offset = k - d;
				odata[k] = idata[k] + idata[offset];
			}
			else {
				odata[k] = idata[k];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			
			int *dev_idata;
			int *dev_odata;

			// smallest power of 2 >= n
			int pow2 = pow(2,ilog2ceil(n));
			cudaMalloc((void**)&dev_idata, (pow2 + 1) * sizeof(int));
			checkCUDAError("cudaMalloc error dev_idata");

			cudaMalloc((void**)&dev_odata, (pow2 + 1) * sizeof(int));
			checkCUDAError("cudaMalloc error dev_odata");

			// TIMER STARTS HERE
			timer().startGpuTimer();

			// shift the input array to the right and pad with a zero.
			int a = 0;
			cudaMemcpy(&dev_idata[0], &a, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy error 0 dev_idata naive");

			cudaMemcpy(&dev_idata[1], idata, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy error dev_idata naive");

			int levels = ilog2ceil(n);
			dim3 fullBlocks((pow2 + blockSize - 1) / blockSize);

			for (int i = 0; i < levels; i++) {
				int d = pow(2, i);
				kernScan << <fullBlocks, blockSize >> > (n, d, dev_odata, dev_idata);
				int *temp = dev_odata;
				dev_odata = dev_idata;
				dev_idata = temp; 
			}

			// TIMER ENDS HERE
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_idata, sizeof(int)*(n), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
