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

		__global__ void kernScan(int d, int *odata, int *idata) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
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
            timer().startGpuTimer();
			
			int *inData;
			int *outData;

			// smallest power of 2 >= n
			int pow2 = pow(2,ilog2ceil(n));
			cudaMalloc((void**)&inData, (pow2 + 1) * sizeof(int));
			cudaMalloc((void**)&outData, (pow2 + 1) * sizeof(int));

			// shift the input array to the right and pad with a zero.
			int a = 0;
			cudaMemcpy(&inData[0], &a, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(&inData[1], idata, sizeof(int)*n, cudaMemcpyHostToDevice);
			cudaThreadSynchronize();

			int levels = ilog2ceil(n);
			dim3 fullBlocks((pow2 + blockSize - 1) / blockSize);

			for (int i = 0; i < levels; i++) {
				int d = pow(2, i);
				kernScan << <fullBlocks, blockSize >> > (d, outData, inData);
				cudaThreadSynchronize();
				int *temp = outData;
				outData = inData;
				inData = temp; 
			}

			cudaMemcpy(odata, inData, sizeof(int)*(n), cudaMemcpyDeviceToHost);
			cudaFree(inData);
			cudaFree(outData);
            timer().endGpuTimer();
        }
    }
}
