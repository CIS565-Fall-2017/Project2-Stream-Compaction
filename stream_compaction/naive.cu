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
		int threadPerBlock = 256;
		int *dev_0, *dev_1;
		// TODO: 
		__global__ void NaiveScan(int d, int *idata, int *odata, int oLength) {
			int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
			if (idx >= oLength) return;
			int flag = 1 << d;
			odata[idx] = idx >= flag ? idata[idx] + idata[idx - flag] : idata[idx];
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata) {
			int layer = ilog2ceil(n);
			int oLength = 1 << layer;
			cudaMalloc((void**)&dev_0, oLength * sizeof(int));
			cudaMalloc((void**)&dev_1, oLength * sizeof(int));
			checkCUDAError("cudaMalloc failed!");
			cudaMemcpy(dev_0, idata, oLength*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy to device failed!");
			int blocknum = oLength / threadPerBlock + 1;

			timer().startGpuTimer();
			for (int d = 0; d < layer; d++) {
				NaiveScan << <blocknum, threadPerBlock >> >(d, dev_0, dev_1, oLength);
				std::swap(dev_0, dev_1);
			}
			timer().endGpuTimer();

			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_0, (n - 1)*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy to host failed!");

			cudaFree(dev_0);
			cudaFree(dev_1);


		}
	}
}
