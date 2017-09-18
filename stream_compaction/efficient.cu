#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		
		__global__ void kernWorkEfficientParallelUpSweep(int N, int step, int *odata) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < N && (index % step == 0))
				odata[index + step - 1] += odata[index + step/2 - 1];
		
		}

		__global__ void kernWorkEfficientParallelDownSweep(int N, int step, int *odata) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < N && (index % step == 0))
			{
				int aIndex = index + step / 2 - 1;
				int bIndex = index + step - 1;

				int t = odata[aIndex];
				odata[aIndex] = odata[bIndex];
				odata[bIndex] += t;
			}

		}

		//Extended version
		__global__ void UpSweep(int *g_idata, int n, int offsetParam)
		{
			extern __shared__ int temp[]; 
			int thid = threadIdx.x;
			int index = 2 * thid;			

			int offset = 1;

			temp[index] = g_idata[ (index + 1) * offsetParam  - 1 + (blockIdx.x * blockDim.x) * 2]; 
			temp[index + 1] = g_idata[ (index + 2) * offsetParam - 1 + (blockIdx.x * blockDim.x) * 2];

			//Up-Sweep (Parallel Reduction)
			for (int d = n >> 1; d > 0; d >>= 1)
			{
				__syncthreads();
				if (thid < d)
				{
					int ai = offset*(index + 1) - 1;
					int bi = offset*(index + 2) - 1;


					temp[bi] += temp[ai];
				}
				offset *= 2;
			}

			__syncthreads();

			g_idata[(index + 1) * offsetParam - 1 + (blockIdx.x * blockDim.x) * 2] = temp[index]; 
			g_idata[(index + 2) * offsetParam - 1 + (blockIdx.x * blockDim.x) * 2] = temp[index + 1];			
		}

		//Extended version
		__global__ void DownSweep(int *g_idata, int n, int offsetParam)
		{
			extern __shared__ int temp[]; 
			int thid = threadIdx.x;
			int index = 2 * thid;		

			int offset = n;

			temp[index] = g_idata[(index + 1) * offsetParam - 1 + (blockIdx.x * blockDim.x)*2]; 
			temp[index + 1] = g_idata[(index + 2) * offsetParam - 1 + (blockIdx.x * blockDim.x) * 2];

			//Down-Sweep
			for (int d = 1; d < n; d *= 2)
			{
				offset >>= 1;

    			__syncthreads();
				 
				if (thid < d )
				{

					int ai = offset*(index + 1) - 1;
					int bi = offset*(index + 2) - 1;


					int t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}				
			}
			__syncthreads();

			g_idata[(index + 1) * offsetParam - 1 + (blockIdx.x * blockDim.x) * 2] = temp[index]; 
			g_idata[(index + 2) * offsetParam - 1 + (blockIdx.x * blockDim.x) * 2] = temp[index + 1];
		}

		__global__ void prescan(int *g_odata, int *g_idata, int n)
		{
			extern __shared__ int temp[];  // allocated on invocation
			int thid = threadIdx.x;
			int offset = 1;
			
			int index = 2 * thid;			

			temp[index] = g_idata[index]; // load input into shared memory
			temp[index + 1] = g_idata[index + 1];
			

			//Up-Sweep (Parallel Reduction)
			for (int d = n >> 1; d > 0; d >>= 1)                  
			{
				__syncthreads();
				if (thid < d)
				{					
					int ai = offset*(index + 1) - 1;
					int bi = offset*(index + 2) - 1;


					temp[bi] += temp[ai];
				}
				offset *= 2;
			}			

			
			//temp[n - 1] = 0;
			// clear the last element
			if (thid == 0)
			{
				temp[n - 1] = 0;
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


					int t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			__syncthreads();
			
			
			g_odata[index] = temp[index]; // write results to device memory
			g_odata[index + 1] = temp[index + 1];

			
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int level = ilog2ceil(n);
			int poweroftwosize = (int)pow(2, level);
		
			int *dev_idata;

			cudaMalloc((void**)&dev_idata, poweroftwosize * sizeof(int));
			
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int blockSize = pow(2, level);
			blockSize = std::min(blockSize, 2048);

			int blockCount = (n + blockSize - 1) / blockSize;

			timer().startGpuTimer();
			
			int offset = 1;

			do
			{
				dim3 fullBlocksPerGrid(blockCount);
				
				UpSweep <<< fullBlocksPerGrid, blockSize / 2, blockSize * sizeof(int) >> > (dev_idata, blockSize, offset);

				if (blockCount == 1)
					blockCount = 0;
				else
				{
					blockSize = blockCount;
					blockCount = 1;
					offset *= 2048;
				}				
			}
			while (blockCount >= 1);
		
			timer().endGpuTimer();

			// clear the last element
			int last = 0;
			cudaMemcpy(&dev_idata[poweroftwosize-1], &last, sizeof(int), cudaMemcpyHostToDevice);

			blockCount = 1;

			timer().startGpuTimer();

			do
			{
				dim3 fullBlocksPerGrid(blockCount);
				DownSweep << < fullBlocksPerGrid, blockSize / 2, blockSize * sizeof(int) >> > (dev_idata, blockSize, offset);
 				blockCount *= blockSize;
				blockSize = 2048;
				offset /= 2048;
			}
			while (blockCount < n);
			
			timer().endGpuTimer();			

			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
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
			
			int *dev_bools;
			int *dev_odata;
			int *dev_idata;
			int *dev_ScanResult;

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_idata, n * sizeof(int));			

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int level = ilog2ceil(n);
			int poweroftwosize = (int)pow(2, level);

			int blockSize = pow(2, level);
			blockSize = std::min(blockSize, 1024);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			cudaMalloc((void**)&dev_ScanResult, poweroftwosize * sizeof(int));
			cudaMalloc((void**)&dev_bools, poweroftwosize * sizeof(int));

			
            // TODO

			timer().startGpuTimer();
			StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >>>(n, dev_bools, dev_idata);

			cudaMemcpy(dev_ScanResult, dev_bools, sizeof(int)*poweroftwosize, cudaMemcpyDeviceToDevice);


			blockSize = pow(2, level);
			blockSize = std::min(blockSize, 2048);
			int blockCount = (n + blockSize - 1) / blockSize;

			int offset = 1;

			do
			{
				dim3 fullBlocksPerGrid(blockCount);

				UpSweep << < fullBlocksPerGrid, blockSize / 2, blockSize * sizeof(int) >> > (dev_ScanResult, blockSize, offset);

				if (blockCount == 1)
					blockCount = 0;
				else
				{
					blockSize = blockCount;
					blockCount = 1;
					offset *= 2048;
				}
			} while (blockCount >= 1);

			timer().endGpuTimer();

			// clear the last element
			int last = 0;
			cudaMemcpy(&dev_ScanResult[poweroftwosize - 1], &last, sizeof(int), cudaMemcpyHostToDevice);

			blockCount = 1;

			timer().startGpuTimer();

			do
			{
				dim3 fullBlocksPerGrid(blockCount);
				DownSweep << < fullBlocksPerGrid, blockSize / 2, blockSize * sizeof(int) >> > (dev_ScanResult, blockSize, offset);
				blockCount *= blockSize;
				blockSize = 2048;
				offset /= 2048;
			} while (blockCount < n);			

			blockSize = pow(2, level);
			blockSize = std::min(blockSize, 1024);
			//fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize >>>(n, dev_odata, dev_idata, dev_bools, dev_ScanResult);
			timer().endGpuTimer();
			


			cudaMemcpy(odata, dev_odata, sizeof(int)*n, cudaMemcpyDeviceToHost);

			int counter;
			cudaMemcpy(&counter, &dev_ScanResult[poweroftwosize - 1], sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_bools);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_ScanResult);
			
            return counter;
        }





		__global__ void kernEarray(int *g_odata, int *g_idata, int n, int digit)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			g_odata[index] = (g_idata[index] >> digit & 0x01) ? 0 : 1;

		}

		__global__ void kernTarray(int *g_odata, int *g_idata, int *g_e, int n)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);			

			int totalFalses = g_e[n - 1] + g_idata[n - 1];

			g_odata[index] = index - g_idata[index] + totalFalses;

		}

		__global__ void kernDarray(int *g_odata, int *e,  int *f, int *t)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			g_odata[index] = e[index] ? f[index] : t[index];

		}

		__global__ void kernLast(int *g_odata, int *g_idata, int *d)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			g_odata[d[index]] = g_idata[index];
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

			temp[2 * n + index] = temp[n + index];
			temp[2 * n + index + 1] = temp[n + index + 1];


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

			int totalFalses = temp[2 * n - 1] + temp[2 * n + n - 1];

			//t array 3n ~ 4n
			temp[3 * n + thid] = thid - temp[2 * n + thid] + totalFalses;

			__syncthreads();

			//d array 4n ~ 5n
			temp[4 * n + thid] = temp[n + thid] ? temp[2 * n + thid] : temp[3 * n + thid];
			__syncthreads();

			g_idata[temp[4 * n + thid]] = temp[thid];

		}

		void radixScan(int n, int *odata, const int *idata)
		{
			int *g_idata;
			int *g_odata;
			int *g_e;
			int *g_f;
			int *g_t;
			int *g_d;

			cudaMalloc((void**)&g_idata, n * sizeof(int));
			cudaMalloc((void**)&g_odata, n * sizeof(int));
			cudaMemcpy(g_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int level = ilog2ceil(n);
			int poweroftwosize = (int)pow(2, level);

			cudaMalloc((void**)&g_e, poweroftwosize * sizeof(int));
			cudaMalloc((void**)&g_f, poweroftwosize * sizeof(int));
			cudaMalloc((void**)&g_t, poweroftwosize * sizeof(int));
			cudaMalloc((void**)&g_d, poweroftwosize * sizeof(int));

			

			for (int i = 0; i < level; i++)
			{
				int blockSize = pow(2, level);
				blockSize = std::min(blockSize, 1024);
				dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

				timer().startGpuTimer();

				kernEarray << <fullBlocksPerGrid, blockSize >> >(g_e, g_idata, n, i);

				timer().endGpuTimer();
				
				cudaMemcpy(g_f, g_e, sizeof(int) * poweroftwosize, cudaMemcpyDeviceToDevice);

				int blockSize2 = pow(2, level);
				blockSize2 = std::min(blockSize2, 2048);
				int blockCount = (n + blockSize2 - 1) / blockSize2;

				int offset = 1;

				timer().startGpuTimer();

				do
				{
					dim3 fullBlocksPerGrid(blockCount);

					UpSweep << < fullBlocksPerGrid, blockSize2 / 2, blockSize2 * sizeof(int) >> > (g_f, blockSize2, offset);

					if (blockCount == 1)
						blockCount = 0;
					else
					{
						blockSize2 = blockCount;
						blockCount = 1;
						offset *= 2048;
					}
				} while (blockCount >= 1);

				timer().endGpuTimer();

				// clear the last element
				int last = 0;
				cudaMemcpy(&g_f[poweroftwosize - 1], &last, sizeof(int), cudaMemcpyHostToDevice);

				blockCount = 1;

				timer().startGpuTimer();

				do
				{
					dim3 fullBlocksPerGrid(blockCount);
					DownSweep << < fullBlocksPerGrid, blockSize2 / 2, blockSize2 * sizeof(int) >> > (g_f, blockSize2, offset);
					blockCount *= blockSize2;
					blockSize2 = 2048;
					offset /= 2048;
				} while (blockCount < n);

				kernTarray << <fullBlocksPerGrid, blockSize >> >(g_t, g_f, g_e, n);
				kernDarray << <fullBlocksPerGrid, blockSize >> >(g_d, g_e, g_f, g_t);
				kernLast << <fullBlocksPerGrid, blockSize >> > (g_odata, g_idata, g_d);

				timer().endGpuTimer();

				cudaMemcpy(g_idata, g_odata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
					
			}

			

			cudaMemcpy(odata, g_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(g_idata);
			cudaFree(g_odata);
			cudaFree(g_e);
			cudaFree(g_t);
			cudaFree(g_f);
			cudaFree(g_d);
		}
    }
}
