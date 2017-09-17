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

		//not using shared memory and kernel iteration
		__global__ void kernNaiveParallelScan(int N, int limit, int *odata, const int *idata) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			
			if(index >= limit)
				odata[index] = idata[index - limit] + idata[index];
			else
				odata[index] = idata[index];
		}

		//using shared memory
		__global__ void kernNaiveParallelSharedScan(int *g_odata, int *g_idata, int n)
		{
			extern __shared__ int temp[];
			
			int thid = threadIdx.x;

			int pout = 0, pin = 1;

			temp[pout*n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
			__syncthreads();

			for (int offset = 1; offset < n; offset *= 2)
			{
				pout = 1 - pout; 
				pin = 1 - pin;
				if (thid >= offset)
					temp[pout*n + thid] = temp[pin*n + thid - offset] + temp[pin*n + thid];
				else
					temp[pout*n + thid] = temp[pin*n + thid];
				__syncthreads();
			}
			g_odata[thid] = temp[pout*n + thid];
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata) {

			//Version 1
			/*
			int *dev_idata_0;
			int *dev_idata_1;

			cudaMalloc((void**)&dev_idata_0, n * sizeof(int));
			cudaMalloc((void**)&dev_idata_1, n * sizeof(int));

			cudaMemcpy(dev_idata_0, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_idata_1, dev_idata_0, sizeof(int) * n, cudaMemcpyDeviceToDevice);

			int level = ilog2ceil(n);
			int blockSize = pow(2, level);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			bool bUse0 = true;

			timer().startGpuTimer();

			// TODO
			for (int d = 1; d <= level; d++)
			{
				kernNaiveParallelScan << < fullBlocksPerGrid, blockSize >> > (n, pow(2, d - 1), bUse0 ? dev_idata_1 : dev_idata_0, bUse0 ? dev_idata_0 : dev_idata_1);
				bUse0 = !bUse0;
			}

			timer().endGpuTimer();

			//Inclusive to Exclusive
			
			cudaMemcpy(&odata[1], bUse0 ? dev_idata_0 : dev_idata_1, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
			int Identity = 0;
			cudaMemcpy(odata, &Identity, sizeof(int), cudaMemcpyHostToHost);
			
			cudaFree(dev_idata_0);
			cudaFree(dev_idata_1);	
			*/

			//Version 2
			int *g_idata;
			int *g_odata;

			cudaMalloc((void**)&g_idata, n * sizeof(int));
			cudaMalloc((void**)&g_odata, n * sizeof(int));

			cudaMemcpy(g_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaMemcpy(g_odata, g_idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);

			int level = ilog2ceil(n);
			int blockSize = pow(2, level);
			blockSize = std::min(blockSize, 1024);

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			timer().startGpuTimer();

			kernNaiveParallelSharedScan <<< fullBlocksPerGrid, blockSize, n * 2 * sizeof(int) >>> (g_odata, g_idata, n);

			timer().endGpuTimer();

			cudaMemcpy(odata, g_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(g_idata);
			cudaFree(g_odata);
		}

		__global__ void kernRadixScan(int *g_idata, int n, int digit)
		{
			extern __shared__ int temp[];
			int thid = threadIdx.x;			

			//i array 0 ~ n
			temp[thid] = g_idata[thid];

			__syncthreads();

			//e array n ~ 2n			
			temp[n + thid] = (temp[thid] >> digit & 0x01) ? 0 : 1;
			
			__syncthreads();

			//Exclusive Scan e
			//f array    2n ~ 3n			

			int index = 2 * thid;

			temp[2*n + index] = temp[n + index]; 
			temp[2*n + index + 1] = temp[n + index + 1];


			int offset = 1;

			//Up-Sweep (Parallel Reduction)
			for (int d = n >> 1; d > 0; d >>= 1)
			{
				__syncthreads();
				if (thid < d)
				{
					int ai = offset*(index + 1) - 1;
					int bi = offset*(index + 2) - 1;


					temp[2 * n + bi] += temp[2 * n + ai];
				}
				offset *= 2;
			}


			//temp[n - 1] = 0;
			// clear the last element
			if (thid == 0)
			{
				temp[2 * n + n - 1] = 0;
			}


			//Down-Sweep
			for (int d = 1; d < n; d *= 2)
			{
				offset >>= 1;
				__syncthreads();
				if (thid < d)
				{

					int ai = offset*(index + 1) - 1;
					int bi = offset*(index + 2) - 1;


					int t = temp[2 * n + ai];
					temp[2 * n + ai] = temp[2 * n + bi];
					temp[2 * n + bi] += t;
				}
			}
			__syncthreads();
			
			int totalFalses = temp[2 * n - 1] + temp[2 * n + n -1];

			//t array 3n ~ 4n
			temp[3 * n + thid] = thid - temp[2 * n + thid] + totalFalses;

			__syncthreads();

			//d array 4n ~ 5n
			temp[4 * n + thid] = temp[n + thid]  ? temp[2 * n + thid] : temp[3 * n + thid];
			__syncthreads();

			g_idata[temp[4 * n + thid]] = temp[thid];

		}

		void sortArray(int n, int *b, int *a)
		{
			std::vector<int> arr;

			for (int i = 0; i < n; i++)
			{
				arr.push_back(a[i]);
			}

			timer().startCpuTimer();
			std::sort(arr.begin(), arr.end());
			timer().endCpuTimer();

			for (int i = 0; i < n; i++)
			{
				b[i] = arr[i];
			}
		}

		void radixScan(int n, int *odata, const int *idata)
		{
			int *g_idata;

			cudaMalloc((void**)&g_idata, n * sizeof(int));
			cudaMemcpy(g_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int level = ilog2ceil(n);
			int blockSize = pow(2, level);
			blockSize = std::min(blockSize, 1024);

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			timer().startGpuTimer();

			for (int i = 0; i < level; i++)
			{
				kernRadixScan <<< fullBlocksPerGrid, blockSize, 5 * n * sizeof(int) >> > (g_idata, n, i);
			}

			timer().endGpuTimer();

			cudaMemcpy(odata, g_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(g_idata);
		}
	}
}
