#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernSetZero(int N, int* dev_data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			dev_data[index] = 0;
		}

		__global__ void kernEffcientUpSweep(int N, int offset, int* dev_data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			// ------------ old branch mehthod----------------
			/*if ((index + 1) % offset == 0) {
				dev_data[index] += dev_data[index - offset / 2];
			}*/
			// -----------------------------------------------

			int targetIndex = (index + 1) * offset - 1;
			dev_data[targetIndex] += dev_data[targetIndex - offset / 2];
		}

		__global__ void kernSetRootZero(int N, int* dev_data) {
			dev_data[N - 1] = 0;
		}

		__global__ void kernEfficientDownSweep(int N, int offset, int* dev_data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}

			// ------------ old branch mehthod----------------
			/*if ((index + 1) % offset == 0) {
				int t = dev_data[index - offset / 2];
				dev_data[index - offset / 2] = dev_data[index];
				dev_data[index] += t;
			}*/
			// -----------------------------------------------

			int targetIndex = (index + 1) * offset - 1;

			int t = dev_data[targetIndex - offset / 2];
			dev_data[targetIndex - offset / 2] = dev_data[targetIndex];
			dev_data[targetIndex] += t;
		}

		__global__ void kernSetCompactCount(int N, int* dev_count, int* bools, int* indices) {
			dev_count[0] = bools[N - 1] ? (indices[N - 1] + 1) : indices[N - 1];
		}

		/// ------------------- EX : Dynamic Shared Memo ----------------------
		__global__ void kernScanDynamicShared(int n, int *g_odata, int *g_idata, int *OriRoot) {
			extern __shared__ int temp[];

			//int index = threadIdx.x + (blockIdx.x * blockDim.x);
			//if (index >= N) {
			//	return;
			//}

			int thid = threadIdx.x;
			// assume it's always a 1D block
			int blockOffset = 2 * blockDim.x * blockIdx.x;
			int offset = 1;

			//temp[2 * thid] = g_idata[blockOffset + 2 * thid];
			//temp[2 * thid + 1] = g_idata[blockOffset + 2 * thid + 1];
			int ai = thid;
			int bi = thid + (n / 2);
			int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
			int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
			temp[ai + bankOffsetA] = g_idata[blockOffset + ai];
			temp[bi + bankOffsetB] = g_idata[blockOffset + bi];



			// UP-sweep
			for (int d = n >> 1; d > 0; d >>= 1) {
				__syncthreads();
				if (thid < d) {
					//int ai = offset * (2 * thid + 1) - 1;
					//int bi = offset * (2 * thid + 2) - 1;
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1; 
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);


					temp[bi] += temp[ai];
				}
				offset *= 2;
			}

			__syncthreads();
			// save origin root and set it to zero
			if (thid == 0) { 
				OriRoot[blockIdx.x] = temp[n - 1];
				//temp[n - 1] = 0;
				temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
			}

			for (int d = 1; d < n; d *= 2) {
				offset >>= 1;
				__syncthreads();
				if (thid < d) {
					//int ai = offset * (2 * thid + 1) - 1;
					//int bi = offset * (2 * thid + 2) - 1;
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);

					int t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			__syncthreads();
			//g_odata[blockOffset + 2 * thid] = temp[2 * thid];
			//g_odata[blockOffset + 2 * thid + 1] = temp[2 * thid + 1];
			g_odata[blockOffset + ai] = temp[ai + bankOffsetA];
			g_odata[blockOffset + bi] = temp[bi + bankOffsetB];
		}

		__global__ void kernAddOriRoot(int N, int* OriRoot, int* dev_odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			dev_odata[index] += OriRoot[blockIdx.x];
		}

		/// -------------------------------------------------------------------

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int dMax = ilog2ceil(n);
			int size = (int)powf(2.0f, (float)dMax);

			int* dev_data;

			cudaMalloc((void**)&dev_data, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaDeviceSynchronize();

			dim3 blockDim(blockSize);
			dim3 gridDim((size + blockSize - 1) / blockSize);

			kernSetZero << < gridDim, blockDim >> > (size, dev_data);
			checkCUDAError("kernSetZero failed!");

			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("efficient cudaMemcpy failed!");


            timer().startGpuTimer();

			// Step 1 : Up-sweep
			for (int d = 0; d <= dMax - 1; d++) {
				// ------------ old branch mehthod----------------
				//kernEffcientUpSweep << <gridDim, blockDim >> > (size, (int)powf(2.0f, (float)d + 1.0f), dev_data);
				// -----------------------------------------------

				//only launch threads that acutally work
				int temp_size = (int)powf(2.0f, (float)(dMax - d - 1));
				kernEffcientUpSweep << <dim3((temp_size + blockSize - 1) / blockSize), blockDim >> > (temp_size, (int)powf(2.0f, (float)d + 1.0f), dev_data);

			}
			checkCUDAError("kernEffcientUpSweep failed!");

			// Step 2 : Down-sweep
			kernSetRootZero << < dim3(1), dim3(1) >> > (size, dev_data);
			checkCUDAError("kernSetRootZero failed!");

			for (int d = dMax - 1; d >= 0; d--) {
				// ------------ old branch mehthod----------------
				//kernEfficientDownSweep << <gridDim, blockDim >> > (size, (int)powf(2.0f, (float)d + 1.0f), dev_data);
				// -----------------------------------------------

				//only launch threads that acutally work
				int temp_size = (int)powf(2.0f, (float)(dMax - d - 1));
				kernEfficientDownSweep << <dim3((temp_size + blockSize - 1) / blockSize), blockDim >> > (temp_size, (int)powf(2.0f, (float)d + 1.0f), dev_data);
			}
			checkCUDAError("kernEfficientDownSweep failed!");

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("efficient cudaMemcpy failed!");
			
			cudaFree(dev_data);
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
			// compact Set-up
			int* dev_idata;
			int* dev_odata;
			int* bools;
			int* indices;
			int* dev_count;
			int count;

			dim3 blockDim(blockSize);
			dim3 gridDim((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&bools, n * sizeof(int));
			checkCUDAError("cudaMalloc bools failed!");
			cudaMalloc((void**)&dev_count, sizeof(int));
			checkCUDAError("cudaMalloc dev_count failed!");
			cudaDeviceSynchronize();

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("efficient compact cudaMemcpy failed!");



			// scan Set-up
			int dMax = ilog2ceil(n);
			int size = (int)powf(2.0f, (float)dMax);

			dim3 scan_gridDim((size + blockSize - 1) / blockSize);

			cudaMalloc((void**)&indices, size * sizeof(int));
			checkCUDAError("cudaMalloc indices failed!");
			cudaDeviceSynchronize();

			kernSetZero << < scan_gridDim, blockDim >> > (size, indices);
			checkCUDAError("kernSetZero failed!");


            timer().startGpuTimer();
			// Step 1 : compute bools array
			StreamCompaction::Common::kernMapToBoolean << <gridDim, blockDim >> > (n, bools, dev_idata);
			checkCUDAError("kernMapToBoolean failed!");

			cudaMemcpy(indices, bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy failed!");

			// Step 2 : exclusive scan indices
			// Up-sweep
			for (int d = 0; d <= dMax - 1; d++) {
				// ------------ old branch mehthod----------------
				//kernEffcientUpSweep << <scan_gridDim, blockDim >> > (size, (int)powf(2.0f, (float)d + 1.0f), indices);
				// -----------------------------------------------

				//only launch threads that acutally work
				int temp_size = (int)powf(2.0f, (float)(dMax - d - 1));
				kernEffcientUpSweep << <dim3((temp_size + blockSize - 1) / blockSize), blockDim >> > (temp_size, (int)powf(2.0f, (float)d + 1.0f), indices);
			}
			checkCUDAError("kernEffcientUpSweep failed!");

			// Down-sweep
			kernSetRootZero << < dim3(1), dim3(1) >> > (size, indices);
			checkCUDAError("kernSetRootZero failed!");

			for (int d = dMax - 1; d >= 0; d--) {
				// ------------ old branch mehthod----------------
				//kernEfficientDownSweep << <scan_gridDim, blockDim >> > (size, (int)powf(2.0f, (float)d + 1.0f), indices);
				// -----------------------------------------------

				//only launch threads that acutally work
				int temp_size = (int)powf(2.0f, (float)(dMax - d - 1));
				kernEfficientDownSweep << <dim3((temp_size + blockSize - 1) / blockSize), blockDim >> > (temp_size, (int)powf(2.0f, (float)d + 1.0f), indices);
			}
			checkCUDAError("kernEfficientDownSweep failed!");

			// Step 3 : Scatter
			StreamCompaction::Common::kernScatter << <gridDim, blockDim >> > (n, dev_odata, dev_idata, bools, indices);
			checkCUDAError("kernScatter failed!");

			kernSetCompactCount << <dim3(1), dim3(1) >> > (n, dev_count, bools, indices);
			checkCUDAError("kernSetCompactCount failed!");

            timer().endGpuTimer();


			cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaMemcpy(odata, dev_odata, sizeof(int) * count, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(bools);
			cudaFree(dev_count);
			cudaFree(indices);

            return count;
        }


		void scanDynamicShared(int n, int *odata, const int *idata) {
			int dMax = ilog2ceil(n);
			int size = (int)powf(2.0f, (float)dMax);

			int* dev_data;
			int* ori_root;

			int dynamicMemoBlockSize = 64;

			dim3 blockDim(dynamicMemoBlockSize);
			dim3 gridDim((size + dynamicMemoBlockSize - 1) / dynamicMemoBlockSize);

		
			cudaMalloc((void**)&dev_data, sizeof(int) * size);
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&ori_root, sizeof(int) * gridDim.x);
			checkCUDAError("cudaMalloc ori_root failed!");
			cudaDeviceSynchronize();

			kernSetZero << < gridDim, blockDim >> > (size, dev_data);
			checkCUDAError("kernSetZero failed!");
			kernSetZero << < dim3((gridDim.x + dynamicMemoBlockSize - 1) / dynamicMemoBlockSize), blockDim >> > (gridDim.x, ori_root);
			checkCUDAError("kernSetZero failed!");

			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("naive cudaMemcpy failed!");

			int sharedMemoryPerBlockInBytes = dynamicMemoBlockSize * sizeof(int); // Compute This

			timer().startGpuTimer();

			//kernScanDynamicShared << <gridDim, blockDim, sharedMemoryPerBlockInBytes >> > (size, dynamicMemoBlockSize, dev_data, dev_data, ori_root);
			kernScanDynamicShared << <gridDim, dim3(blockDim.x / 2), sharedMemoryPerBlockInBytes >> > (dynamicMemoBlockSize, dev_data, dev_data, ori_root);


			sharedMemoryPerBlockInBytes = gridDim.x * sizeof(int);
			// We only do scan of scan ONE time here
			// Actually, it should be a while loop here
			// This process should happen until root number we get < blockDim.x
			//kernScanDynamicShared << < dim3(1), dim3(gridDim.x), sharedMemoryPerBlockInBytes >> > (gridDim.x, gridDim.x, ori_root, ori_root, ori_root);
			kernScanDynamicShared << < dim3(1), dim3(gridDim.x / 2), sharedMemoryPerBlockInBytes >> > (gridDim.x, ori_root, ori_root, ori_root);

			kernAddOriRoot << <gridDim, blockDim >> > (size, ori_root, dev_data);

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaFree(dev_data);
			cudaFree(ori_root);
		}


		int compactDynamicShared(int n, int *odata, const int *idata) {
			// compact Set-up
			int* dev_idata;
			int* dev_odata;
			int* bools;
			int* indices;
			int* dev_count;
			int count;

			dim3 blockDim(blockSize);
			dim3 gridDim((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&bools, n * sizeof(int));
			checkCUDAError("cudaMalloc bools failed!");
			cudaMalloc((void**)&dev_count, sizeof(int));
			checkCUDAError("cudaMalloc dev_count failed!");
			cudaDeviceSynchronize();

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("efficient compact cudaMemcpy failed!");


			// scan Set-up
			int dMax = ilog2ceil(n);
			int size = (int)powf(2.0f, (float)dMax);

			int* ori_root;

			dim3 scan_gridDim((size + blockSize - 1) / blockSize);

			cudaMalloc((void**)&indices, size * sizeof(int));
			checkCUDAError("cudaMalloc indices failed!");
			cudaMalloc((void**)&ori_root, sizeof(int) * gridDim.x);
			checkCUDAError("cudaMalloc ori_root failed!");
			cudaDeviceSynchronize();

			kernSetZero << < scan_gridDim, blockDim >> > (size, indices);
			checkCUDAError("kernSetZero failed!");
			kernSetZero << < dim3((scan_gridDim.x + gridDim.x - 1) / gridDim.x), blockDim >> > (gridDim.x, ori_root);
			checkCUDAError("kernSetZero failed!");

			int sharedMemoryPerBlockInBytes = blockDim.x * sizeof(int); // Compute This

			timer().startGpuTimer();

			// Step 1 : compute bools array
			StreamCompaction::Common::kernMapToBoolean << <gridDim, blockDim >> > (n, bools, dev_idata);
			checkCUDAError("kernMapToBoolean failed!");

			cudaMemcpy(indices, bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy failed!");

			// Step 2 : exclusive scan indices
			//kernScanDynamicShared << <scan_gridDim, blockDim, sharedMemoryPerBlockInBytes >> > (size, blockDim.x, indices, indices, ori_root);
			kernScanDynamicShared << <scan_gridDim, dim3(blockDim.x / 2), sharedMemoryPerBlockInBytes >> > (blockDim.x, indices, indices, ori_root);

			sharedMemoryPerBlockInBytes = gridDim.x * sizeof(int);
			// We only do scan of scan ONE time here
			// Actually, it should be a while loop here
			// This process should happen until root number we get < blockDim.x
			//kernScanDynamicShared << < dim3(1), dim3(gridDim.x), sharedMemoryPerBlockInBytes >> > (scan_gridDim.x, scan_gridDim.x, ori_root, ori_root, ori_root);
			kernScanDynamicShared << < dim3(1), dim3(gridDim.x / 2), sharedMemoryPerBlockInBytes >> > (scan_gridDim.x, ori_root, ori_root, ori_root);
			kernAddOriRoot << <scan_gridDim, blockDim >> > (size, ori_root, indices);


			// Step 3 : Scatter
			StreamCompaction::Common::kernScatter << <gridDim, blockDim >> > (n, dev_odata, dev_idata, bools, indices);
			checkCUDAError("kernScatter failed!");

			kernSetCompactCount << <dim3(1), dim3(1) >> > (n, dev_count, bools, indices);
			checkCUDAError("kernSetCompactCount failed!");

			timer().endGpuTimer();


			cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");
			cudaMemcpy(odata, dev_odata, sizeof(int) * count, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(bools);
			cudaFree(dev_count);
			cudaFree(indices);
			cudaFree(ori_root);

			return count;
		}
    }
}
