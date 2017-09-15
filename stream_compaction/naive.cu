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

		__global__ void kernNaiveScan(int n, int i, int *dev_odata, int *dev_idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			
			int power = 1 << (i - 1);
			dev_odata[index] = (index >= power) ? dev_idata[index - power] + dev_idata[index] : dev_idata[index];
		}

		__global__ void kernShiftRight(int n, int *dev_odata, int *dev_idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}

			dev_odata[index] = (index == 0) ? 0 : dev_idata[index - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int size = n * sizeof(int);

			// Allocate buffers
			int *dev_odata, *dev_idata;
			cudaMalloc((void**)&dev_odata, size);
			checkCUDAError("cudaMalloc dev_odata failed", __LINE__);

			cudaMalloc((void**)&dev_idata, size);
			checkCUDAError("cudaMalloc dev_idata failed", __LINE__);

			// Copy input to device
			cudaMemcpy(dev_idata, idata, size, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata failed", __LINE__);

			// Call kernel
            timer().startGpuTimer();
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);

			int iterations = ilog2ceil(n);
			for (int i = 1; i <= iterations; i++) {
				kernNaiveScan<<<blocksPerGrid, threadsPerBlock>>>(n, i, dev_odata, dev_idata);
				checkCUDAError("kernNaiveScan failed", __LINE__);

				// Swap buffers
				int *temp = dev_odata;
				dev_odata = dev_idata;
				dev_idata = temp;
			}

			kernShiftRight<<<blocksPerGrid, threadsPerBlock>>>(n, dev_odata, dev_idata);
			timer().endGpuTimer();

			// Copy output from device
			cudaMemcpy(odata, dev_odata, size, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata failed", __LINE__);

			// Free buffers
			cudaFree(dev_odata);
			cudaFree(dev_idata);
        }
    }
}
