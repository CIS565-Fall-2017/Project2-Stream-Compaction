#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 896
namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		// TODO: __global__
		__global__ void work(const int n, int *idata, int *odata) {

			extern __shared__ int temp[];
			int index = blockIdx.x * blockDim.x + threadIdx.x;

			if (index >= n) {
				return;
			}
			/*
			if (index >= n) {
				return;
			}

			if (index - step < 0) {
				odata[index] = idata[index];
			} else {
				odata[index] = idata[index] + idata[index - step];
			}
			*/

			int input = 1, output = 0;
			temp[index] = idata[index];
			__syncthreads();
			//33 43 45
			//0 33 43 45
			//0 33 76 88
			//0 33 76 121
			for (int step = 1; step < n; step <<= 1) {
				input ^= 1;
				output ^= 1;
				if (index - step < 0) {
					temp[output * n + index] = temp[input * n + index];
				}
				else {
					temp[output * n + index] = temp[input * n + index] + temp[input * n + index - step];
				}
				__syncthreads();
			}
			odata[index] = temp[(output*n) + index];
		}

		__global__ void moveToExclusive(const int n, int *idata, int *odata) {

			int index = blockIdx.x * blockDim.x + threadIdx.x;

			if (index >= n) {
				return;
			}
			else if (index == 0) {
				odata[index] = 0;
				return;
			}

			odata[index] = idata[index - 1];
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
//            timer().startGpuTimer();
			int *dev_input;
			int *dev_output;
			int input = 1;
			int output = 0;
			// device memory allocation
			cudaMalloc((void**)&dev_input, sizeof(int) * n);
			cudaMalloc((void**)&dev_output, sizeof(int) * n);
			cudaMemcpy(dev_output, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			dim3 blockCount = (n - 1) / BLOCK_SIZE + 1;
			moveToExclusive << <blockCount, BLOCK_SIZE >> >(n, dev_output, dev_input);
			//We want exclusive result. Not inclusive.
			timer().startGpuTimer();
			work << <blockCount, BLOCK_SIZE,2 * n * sizeof(int) >> >(n, dev_input, dev_output);
			timer().endGpuTimer();
			cudaDeviceSynchronize();
			cudaMemcpy(odata, dev_output, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_input);
			cudaFree(dev_output);
//            timer().endGpuTimer();
        }
    }
}
