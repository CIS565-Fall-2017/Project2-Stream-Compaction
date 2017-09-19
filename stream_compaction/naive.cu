#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 896


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		int *dev_in, *dev_out;
		
		__global__ void kernScan(int n, int *dev_in, int *dev_out) {
			/*
			//=============================
			//====without shared memory====
			//=============================
			int *temp;
			int index = threadIdx.x;
			if (index >= n)
				return;
			dev_out[index] = (index > 0) ? dev_in[index - 1] : 0;
			__syncthreads();
			for (int offset = 1; offset < n; offset *= 2) {
				temp = dev_in;
				dev_in = dev_out;
				dev_out = temp;
				if (index >= offset)
					dev_out[index] = dev_in[index] + dev_in[index - offset];
				else
					dev_out[index] = dev_in[index];
				__syncthreads();
			}
			*/
			
			//==============================
			//======with shared memory======
			//==============================
			extern __shared__ int temp[];
			
			int pout = 0, pin = 1;
			int index = threadIdx.x;
			if (index >= n)
				return;
			temp[pout * n + index] = (index > 0) ? dev_in[index - 1] : 0;
			__syncthreads();
			
			for (int offset = 1; offset < n; offset *= 2) {
				pout = 1 - pout;
				pin = 1 - pout;
				if (index >= offset)
					temp[pout * n + index] = temp[pin * n + index] + temp[pin * n + index - offset];
				else
					temp[pout * n + index] = temp[pin * n + index];
				__syncthreads();
			}
			dev_out[index] = temp[pout * n + index];
			
			
			
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		void scan(int n, int *odata, const int *idata) {
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			cudaMalloc((void**)&dev_out, n * sizeof(int));

			cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();
            // TODO
			
			kernScan << < fullBlocksPerGrid, blockSize, 2 * n * sizeof(int) >>> (n, dev_in, dev_out);
			
            timer().endGpuTimer();

			cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);

			cudaFree(dev_in);
			cudaFree(dev_out);
        }
    }
}
