#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCKSIZE 512

namespace StreamCompaction {
    namespace Naive {

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
		__global__ void scan(const int n, const int _d, int * idata, int * odata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}

			odata[idx] = idx >= _d ? idata[idx - _d] + idata[idx] : idata[idx];

		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
			int dSize, dLen;
			int *dev_idata, *dev_odata;

			dLen = 1 << ilog2ceil(n);
			dSize = dLen * sizeof(int);

			dim3 blocksPerGrid((dLen + BLOCKSIZE - 1) / BLOCKSIZE);
			dim3 threadsPerBlocks(BLOCKSIZE);

			cudaMalloc((void**)&dev_idata, dSize);
			cudaMalloc((void**)&dev_odata, dSize);

			cudaMemcpy(dev_idata, idata, dSize, cudaMemcpyHostToDevice);

			for (int _d = 1; _d < dLen; _d <<= 1) {
				scan <<<blocksPerGrid, threadsPerBlocks >>>(n, _d, dev_idata, dev_odata);
				std::swap(dev_idata, dev_odata);
			}

			cudaMemcpy(odata + 1, dev_idata, dSize - sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;			

			cudaFree(dev_idata);
			cudaFree(dev_odata);

            timer().endGpuTimer();
        }
    }
}
