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
		__global__ void kernNaiveScanPass(int n, int pass, const int* idata, int* odata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= n) return;
			int pow = 1 << (pass - 1);
			if (idx >= pow) {
				odata[idx] = idata[idx] + idata[idx - pow];
			}
			else odata[idx] = idata[idx];
		}

		__global__ void kernShiftRight(int n, int* idata, int* odata)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= n) return;

			if (idx == 0) odata[0] = 0;
			else odata[idx] = idata[idx - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // allocate cuda arrays. copy idata to array.
			int* dev_scan_in;
			int* dev_scan_out;

			cudaMalloc((void**)&dev_scan_in, n * sizeof(int));
			checkCUDAError("scan naive in allocation failed");

			cudaMalloc((void**)&dev_scan_out, n * sizeof(int));
			checkCUDAError("scan naive out allocation failed");

			cudaMemcpy(dev_scan_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("scan naive memcpy failed");
			int blockSize = 128;
			dim3 arrCountSize((n + blockSize - 1) / blockSize);

			timer().startGpuTimer();

			int lvls = ilog2ceil(n);
			for (int i = 1; i <= lvls; i++) {
				kernNaiveScanPass << <arrCountSize, blockSize >> >(n, i, dev_scan_in, dev_scan_out);

				// ping pong buffers
				int* t = dev_scan_in;
				dev_scan_in = dev_scan_out;
				dev_scan_out = t;
			}

			kernShiftRight << < arrCountSize, blockSize >> > (n, dev_scan_in, dev_scan_out);

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_scan_out, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_scan_in);
			cudaFree(dev_scan_out);
        }
    }
}
