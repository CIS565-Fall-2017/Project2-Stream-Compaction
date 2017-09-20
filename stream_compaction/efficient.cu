#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		int threadPerBlock = 256;
		int* dev_Data;
		int *dev_Map;
		int *dev_Scatter;
		int *dev_oData;
		int *dev_total;

		__global__ void KernUpSweep(int d, int *idata, int nodeNum)
		{
			int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (idx >= nodeNum)	return;
			idata[(idx + 1)*(1 << (d + 1)) - 1] += idata[idx*(1 << (d + 1)) + (1 << d) - 1];
		}

		__global__ void KernDownSweep(int d, int *idata, int nodeNum)
		{
			int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (idx >= nodeNum)	return;
			int nodeIdx = idx*(1 << (d + 1)) + (1 << d) - 1;
			int temp = idata[nodeIdx];
			idata[nodeIdx] = idata[nodeIdx + (1 << d)];
			idata[nodeIdx + (1 << d)] += temp;
		}
		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata) {
			int layer = ilog2ceil(n);
			int oLength = 1 << layer;
			cudaMalloc((void**)&dev_Data, oLength * sizeof(int));
			checkCUDAError("cudaMalloc failed!");
			cudaMemcpy(dev_Data, idata, sizeof(int) * oLength, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy to device failed!");

			timer().startGpuTimer();
			for (int d = 0; d < layer; d++)
			{
				int nodeNum = 1 << (layer - 1 - d);
				int blocknum = nodeNum / threadPerBlock + 1;
				KernUpSweep << <blocknum, threadPerBlock >> >(d, dev_Data, nodeNum);
			}
			cudaMemset(dev_Data + oLength - 1, 0, sizeof(int));
			checkCUDAError("cudaMemset failed!");
			for (int d = layer - 1; d >= 0; d--)
			{
				int nodeNum = 1 << (layer - 1 - d);
				int blocknum = nodeNum / threadPerBlock + 1;
				KernDownSweep << <blocknum, threadPerBlock >> >(d, dev_Data, nodeNum);
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_Data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy to host failed!");
			cudaFree(dev_Data);

		}

		/**
		* Performs stream compaction on idata, storing the result into odata.
		* All zeroes are discarded.
		*
		* @param n      The number of elements in idata.
		* @param odata  The array into which to store elements.
		* @param idata  The array of elements to compact.
		* @returns      The number of elements remaining after compaction.
		*/
		int compact(int n, int *odata, const int *idata) {
			if (n <= 0)	return -1;
			int layer = ilog2ceil(n);
			int oLength = 1 << layer;
			cudaMalloc((void**)&dev_Data, oLength * sizeof(int));
			cudaMalloc((void**)&dev_Scatter, oLength * sizeof(int));
			cudaMalloc((void**)&dev_Map, oLength * sizeof(int));
			cudaMalloc((void**)&dev_oData, n * sizeof(int));
			checkCUDAError("cudaMalloc failed!");
			cudaMemcpy(dev_Data, idata, oLength * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy to device failed!");


			// TODO
			int blocknum = oLength / threadPerBlock + 1;
			timer().startGpuTimer();
			Common::kernMapToBoolean << <blocknum, threadPerBlock >> >(oLength, dev_Map, dev_Data);

			// Here I reimplement the scan part, because in the main function, scan and compaction are timed seperately,
			// and I don't want to allocate memory for data 2 times.
			cudaMemcpy(dev_Scatter, dev_Map, oLength * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy device to device failed!");

			for (int d = 0; d < layer; d++)
			{
				int nodeNum = 1 << (layer - 1 - d);
				blocknum = nodeNum / threadPerBlock + 1;
				KernUpSweep << <blocknum, threadPerBlock >> >(d, dev_Scatter, nodeNum);
			}

			cudaMemset(dev_Scatter + oLength - 1, 0, sizeof(int));
			checkCUDAError("cudaMemcpy to device failed!");
			for (int d = layer - 1; d >= 0; d--)
			{
				int nodeNum = 1 << (layer - 1 - d);
				blocknum = nodeNum / threadPerBlock + 1;
				KernDownSweep << <blocknum, threadPerBlock >> >(d, dev_Scatter, nodeNum);
			}

			blocknum = n / threadPerBlock + 1;
			Common::kernScatter << < blocknum, threadPerBlock >> > (n, dev_oData, dev_Data, dev_Map, dev_Scatter);
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_oData, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy to host failed!");
			int count, end;
			cudaMemcpy(&count, dev_Scatter + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&end, dev_Map + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy device to device failed!");
			cudaFree(dev_Data);
			cudaFree(dev_Scatter);
			cudaFree(dev_Map);
			cudaFree(dev_oData);

			return end ? count + 1 : count;
		}



	}
}
