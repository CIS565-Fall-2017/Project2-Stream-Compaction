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

		__global__ void naive_sum(int n,int* odata, int* idata)
		{		
			int index = blockIdx.x *blockDim.x + threadIdx.x;
			int i;
			int * ping;
			int * pong;
			int * swap;
			ping = idata;
			pong = odata;
			__syncthreads();

			for (i = 1; i < n; i *= 2)
			{
				if (index - i >= 0)
				{
					pong[index] = ping[index] + ping[index - i];
				}
				else
				{
					pong[index] = ping[index];
				}
				//Ping-Pong here!
				swap = ping;
				ping = pong;
				pong = swap;
				__syncthreads();
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
			int count;


			//Create 2 buffers
			int* buffer1;
			int* buffer2;
			cudaMalloc((void**)&buffer1, n * sizeof(int));
			cudaMalloc((void**)&buffer2, n * sizeof(int));
			cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to buffer1 in GPU
			cudaMemcpy(buffer2, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to buffer2 in GPU
			//Start timer
			timer().startGpuTimer();
			//Do Naive_Sum here! 
			naive_sum << <1,n>> > (n, buffer2, buffer1);
			//End timer
			timer().endGpuTimer();
			//COPY data back to CPU
			cudaMemcpy(odata, buffer1, n * sizeof(int), cudaMemcpyDeviceToHost);
			//Free buffers
			cudaFree(buffer1);
			cudaFree(buffer2);

        }
    }
}
