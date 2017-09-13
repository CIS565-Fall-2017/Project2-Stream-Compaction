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
        // TODO
		__global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) 
		{
			int index = threadIdx.x;
			if (index >= n) return;

			int val = 1 << (d - 1);
			for (int k = val; k < n; ++k) {
				odata[k] = odata[k] + odata[k - val];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            // TODO
			for (int i = 0; i < n; ++i) odata[i] = idata[i];
			for (int d = 1; d < ilog2ceil(n); ++d) {
				kernNaiveScan << <1, n >> > (n, d, odata, idata);
			}
			checkCUDAError("kernComputeIndices failed!");

            timer().endGpuTimer();
        }
    }
}
