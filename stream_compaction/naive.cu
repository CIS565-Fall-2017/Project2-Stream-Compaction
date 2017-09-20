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
		int threadPerBlock = 512;
		int *dev_0, *dev_1;
		// TODO: 
		__global__ void NaiveScan(int d, int *idata, int *odata, int oLength) {
			int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
			if (idx >= oLength) return;
			int flag = 1 << d;
			odata[idx] = idx >= flag ? idata[idx] + idata[idx - flag] : idata[idx];
		}
		//int threadPerBlock = 1024;
		//int BlockNum;

		//int *dev_Data[2];

		//__global__ void CudaScan(int d, int *in, int *out, int n)
		//{
		//	int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
		//	if (thid >= n)
		//		return;
		//	int m = 1 << (d - 1);

		//	if (thid >= m)
		//		out[thid] = in[thid] + in[thid - m];
		//	else
		//		out[thid] = in[thid];

		//}

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



			/*int nCeilLog = ilog2ceil(n);
			int nLength = 1 << nCeilLog;

			cudaMalloc((void**)&dev_Data[0], nLength * sizeof(int));
			cudaMalloc((void**)&dev_Data[1], nLength * sizeof(int));
			checkCUDAError("cudaMalloc failed!");

			cudaMemcpy(dev_Data[0], idata, sizeof(int) * nLength, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy to device failed!");
			int nOutputIndex = 0;*/
			timer().startGpuTimer();
			for (int d = 0; d < layer; d++) {
				NaiveScan << <blocknum, threadPerBlock >> >(d, dev_0, dev_1, oLength);
				std::swap(dev_0, dev_1);
			}
			/*for (int i = 1; i <= nCeilLog; i++)
			{
			nOutputIndex ^= 1;
			BlockNum = nLength / threadPerBlock + 1;
			CudaScan << <BlockNum, threadPerBlock >> >(i, dev_Data[nOutputIndex ^ 1], dev_Data[nOutputIndex], nLength);
			}*/

			timer().endGpuTimer();
			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_0, (n - 1)*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy to host failed!");

			cudaFree(dev_0);
			cudaFree(dev_1);


			/*odata[0] = 0;
			cudaMemcpy(odata + 1, dev_Data[nOutputIndex], sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy to host failed!");

			cudaFree(dev_Data[0]);
			cudaFree(dev_Data[1]);*/


		}
	}
}
