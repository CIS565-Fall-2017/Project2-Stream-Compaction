#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
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
		__global__ void NaiveScanAlgorithm(int* idata,int* odata, int n,int step)
		{
			int index = (blockIdx.x * blockDim.x)+  threadIdx.x;
			
			if ((index >= n) || (index < 0))
			{
				return;
			}			
			if (index >= step)
			{
				odata[index] = idata[index - step] + idata[index];
			}
			else
			{
				odata[index] = idata[index];
			}
					    
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			//Linearlize the arrangement of bolcks
			int blockSize = 256;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			const int memoryCopySize = n * sizeof(int);
			int step;

			//false means output is odata, true means output is idata
			//bool outAndInFlag = false;

			int* dev_idata;
			int* dev_odata;
			int* temp_odata;

			temp_odata = (int*)malloc(memoryCopySize);

			cudaMalloc((void**)&dev_idata, memoryCopySize);
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_idata, idata, memoryCopySize, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");

			cudaMalloc((void**)&dev_odata, memoryCopySize);
			checkCUDAError("cudaMalloc dev_odata failed!");

			for (int d = 1;d <= ilog2ceil(n);d++)
			{
				step = pow(2, d - 1);
				NaiveScanAlgorithm << <fullBlocksPerGrid, blockSize >> > (dev_idata, dev_odata, n, step);
				cudaThreadSynchronize();
				cudaMemcpy(dev_idata, dev_odata, memoryCopySize, cudaMemcpyDeviceToDevice);
				cudaDeviceSynchronize();
			}
			
			cudaMemcpy(temp_odata, dev_idata, memoryCopySize, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			odata[0] = 0;
			for (int i = 1;i < n;i++)
			{
				odata[i] = temp_odata[i-1];
			}

			cudaFree(dev_idata);
			cudaFree(dev_odata);

			free(temp_odata);
			
            timer().endGpuTimer();
        }
    }
}
