#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define identity 0

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kernGenExclusiveScanFromInclusiveScan(int N, int* dev_odata, int* dev_idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			
			// shift right
			if (index == 0) {
				dev_odata[index] = identity;
			}
			else {
				dev_odata[index] = dev_idata[index - 1];
			}
		}

		__global__ void kernNaiveParallelScan(int N, int d, int* dev_odata, int* dev_idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			int offset = (int)(powf(2.0f, (float)d - 1.0f));

			if (index >= offset) {
				dev_odata[index] = dev_idata[index - offset] + dev_idata[index];
			}
			else {
				dev_odata[index] = dev_idata[index];
			}
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
			int* dev_idata;
			int* dev_odata;
			int* temp;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaDeviceSynchronize();

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("naive cudaMemcpy failed!");

			dim3 blockDim(blockSize);
			dim3 gridDim((n + blockSize- 1) / blockSize);
			int dMax = ilog2ceil(n);

			timer().startGpuTimer();

            // TODO
			for (int d = 1; d <= dMax; d++) {
				// call cuda here
				// PAY ATTENTION : this is an inclusive scan 
				kernNaiveParallelScan << <gridDim, blockDim >> > (n, d, dev_odata, dev_idata);
				checkCUDAError("kernNaiveParallelScan failed!");

				// swap input & output buffer
				temp = dev_idata;
				dev_idata = dev_odata;
				dev_odata = temp;
			}
			// generate exclusive result from inclusive
			kernGenExclusiveScanFromInclusiveScan << <gridDim, blockDim >> > (n, dev_odata, dev_idata);
			checkCUDAError("kernGenExclusiveScanFromInclusiveScan failed!");

            timer().endGpuTimer();

			// copy from dev_odata to host odata
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("naive cudaMemcpy failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
