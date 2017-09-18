#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define identity 0

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kernGenExclusiveScanFromInclusiveScan(int N, int* dev_odata, int* dev_idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			
			// shift right
			if (index == 0) {
				dev_odata[index] = identity;
			}
			else {
				dev_odata[index] = dev_idata[index - 1];
			}
		}

		__global__ void kernNaiveParallelScan(int N, int d, int* dev_odata, int* dev_idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			int offset = (int)(powf(2.0f, (float)d - 1.0f));

			if (index >= offset) {
				dev_odata[index] = dev_idata[index - offset] + dev_idata[index];
			}
			else {
				dev_odata[index] = dev_idata[index];
			}
		}

		/// ------------------- EX : Dynamic Shared Memo ----------------------
		__global__ void kernScanDynamicShared(int N, int n, int *g_odata, int *g_idata, int *OriRoot) {
			extern __shared__ int temp[];

			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			int thid = threadIdx.x;
			int pout = 0, pin = 1;
			// assume it's always a 1D block
			int blockOffset = blockDim.x * blockIdx.x;

			temp[pout * n + thid] =  (thid > 0) ? g_idata[blockOffset + thid - 1] : 0;
			
			__syncthreads();

			for (int offset = 1; offset < n; offset *= 2) {
				pout = 1 - pout;
				pin = 1 - pin;
				if (thid >= offset) {
					temp[pout*n + thid] = temp[pin*n + thid] + temp[pin*n + thid - offset];
				}
				else {
					temp[pout*n + thid] = temp[pin*n + thid];
				}
				__syncthreads();
			}

			if (thid == blockDim.x - 1) {
				OriRoot[blockIdx.x] = g_idata[blockOffset + thid] + temp[pout*n + thid];
			}

			g_odata[blockOffset + thid] = temp[pout*n + thid];	
		}

		__global__ void kernAddOriRoot(int N, int* OriRoot, int* dev_odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			dev_odata[index] += OriRoot[blockIdx.x];
		}

		__global__ void kernSetZero(int N, int* dev_data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			dev_data[index] = 0;
		}

		/// -------------------------------------------------------------------

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
			int* dev_idata;
			int* dev_odata;
			int* temp;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaDeviceSynchronize();

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("naive cudaMemcpy failed!");

			dim3 blockDim(blockSize);
			dim3 gridDim((n + blockSize- 1) / blockSize);
			int dMax = ilog2ceil(n);

			timer().startGpuTimer();

            // TODO
			for (int d = 1; d <= dMax; d++) {
				// call cuda here
				// PAY ATTENTION : this is an inclusive scan 
				kernNaiveParallelScan << <gridDim, blockDim >> > (n, d, dev_odata, dev_idata);
				checkCUDAError("kernNaiveParallelScan failed!");

				// swap input & output buffer
				temp = dev_idata;
				dev_idata = dev_odata;
				dev_odata = temp;
			}
			// generate exclusive result from inclusive
			kernGenExclusiveScanFromInclusiveScan << <gridDim, blockDim >> > (n, dev_odata, dev_idata);
			checkCUDAError("kernGenExclusiveScanFromInclusiveScan failed!");

            timer().endGpuTimer();

			// copy from dev_odata to host odata
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("naive cudaMemcpy failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }

		void scanDynamicShared(int n, int *odata, const int *idata) {

			int dMax = ilog2ceil(n);
			int size = (int)powf(2.0f, (float)dMax);

			int* dev_idata;
			int* dev_odata;
			int* ori_root;

			int dynamicMemoBlockSize = 64;

			dim3 blockDim(dynamicMemoBlockSize);
			dim3 gridDim((size + dynamicMemoBlockSize - 1) / dynamicMemoBlockSize);


			cudaMalloc((void**)&dev_idata, sizeof(int) * size);
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, sizeof(int) * size);
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&ori_root, sizeof(int) * gridDim.x);
			checkCUDAError("cudaMalloc ori_root failed!");
			cudaDeviceSynchronize();

			kernSetZero << < gridDim, blockDim >> > (size, dev_idata);
			checkCUDAError("kernSetZero failed!");
			kernSetZero << < dim3((gridDim.x + dynamicMemoBlockSize - 1) / dynamicMemoBlockSize), blockDim >> > (gridDim.x, ori_root);
			checkCUDAError("kernSetZero failed!");

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("naive cudaMemcpy failed!");

			int sharedMemoryPerBlockInBytes = 2 * dynamicMemoBlockSize * sizeof(int); // Compute This

			timer().startGpuTimer();

			kernScanDynamicShared << <gridDim, blockDim, sharedMemoryPerBlockInBytes >> > (size, dynamicMemoBlockSize, dev_odata, dev_idata, ori_root);

			// TODO : 
			// Only support maximum size blockSize * blockSize = 64 * 64 = 4096 number support now
			// and we only scan origin root one time.
			sharedMemoryPerBlockInBytes = 2 * gridDim.x * sizeof(int);
			kernScanDynamicShared << < dim3(1), gridDim, sharedMemoryPerBlockInBytes >> > (gridDim.x, gridDim.x, ori_root, ori_root, ori_root);
			kernAddOriRoot << <gridDim, blockDim >> > (size, ori_root, dev_odata);

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(ori_root);
		}
    }
}
