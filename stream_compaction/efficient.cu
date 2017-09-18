#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

static int blockSize = 1024;
static dim3 blockNum;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
//Non-optimized Scan:
		__global__ void non_opt_cudaSweepUp(int n, int d, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int interval_length = 1 << (d + 1);
			if (index >= n)
				return;
			if (index % interval_length == 0) {
				data[index + (1 << (d + 1)) - 1] += data[index + (1 << d) - 1];
			}
		}

		__global__ void non_opt_cudaSweepDown(int n, int d, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int interval_length = 1 << (d + 1);
			// k from 0 to n-1
			if (index >= n)
				return;
			if (index % interval_length == 0) {
				int temp = data[index + (1 << d) - 1];
				data[index + (1 << d) - 1] = data[index + (1 << (d + 1)) - 1];
				data[index + (1 << (d + 1)) - 1] += temp;
			}
		}

		void non_opt_scan(int n, int *odata, const int *idata) {
			// TODO
			if (n <= 0)
				return;
			int celllog = ilog2ceil(n);

			int pow2len = 1 << celllog;

			int *dev_data;
			cudaMalloc((void**)&dev_data, pow2len * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");

			cudaMemcpy(dev_data, idata, pow2len * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");

			timer().startGpuTimer();

			//Up-Sweep
			for (int d = 0; d <= celllog - 1; d++) {
				blockNum = (pow2len + blockSize) / blockSize;
				non_opt_cudaSweepUp << <blockNum, blockSize >> >(pow2len, d, dev_data);
			}

			//cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			//Down-Sweep
			cudaMemset(dev_data + pow2len - 1, 0, sizeof(int));
			checkCUDAError("cudaMemset failed!");

			for (int d = celllog - 1; d >= 0; d--) {
				blockNum = (pow2len + blockSize) / blockSize;
				non_opt_cudaSweepDown << <blockNum, blockSize >> >(pow2len, d, dev_data);
			}
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			checkCUDAError("cudaMalloc dev_data to odata failed!");

			cudaFree(dev_data);
			checkCUDAError("cudaFree dev_data failed!");

		}

//Optimized Scan
		__global__ void cudaSweepUp(int n, int d, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);		
			int interval_length = 1 << (d + 1);
			if (index >= n)
				return;
			//int idx1 = index * interval_length + (1 << (d + 1)) - 1;
			//int idx2 = index * interval_length + (1 << d) - 1;
			data[index * interval_length + (1 << (d + 1)) - 1] += data[index * interval_length + (1 << d) - 1];
		}

		__global__ void cudaSweepDown(int n, int d, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int interval_length = 1 << (d + 1);
			// k from 0 to n-1
			if (index >= n)
				return;

			int temp = data[index * interval_length + (1 << d) - 1];
			data[index * interval_length + (1 << d) - 1] = data[index * interval_length + (1 << (d + 1)) - 1];
			data[index * interval_length + (1 << (d + 1)) - 1] += temp;
		}

        void scan(int n, int *odata, const int *idata) {
            // TODO
			if (n <= 0)
				return;
			int celllog = ilog2ceil(n);

			int pow2len = 1 << celllog;

			int *dev_data;
			
			cudaMalloc((void**)&dev_data, pow2len * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");

			cudaMemcpy(dev_data, idata, pow2len * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy failed!");
			
			timer().startGpuTimer();

			//Up-Sweep
			for (int d = 0; d <= celllog - 1; d++) {
				int interval_length = (1 << (d + 1));
				blockNum = (pow2len / interval_length + blockSize) / blockSize;
				cudaSweepUp<<<blockNum, blockSize>>>(pow2len / interval_length, d, dev_data);
			}
			
			//cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			//Down-Sweep
			cudaMemset(dev_data + pow2len - 1, 0, sizeof(int));
			checkCUDAError("cudaMemset failed!");

			for (int d = celllog - 1; d >= 0; d--) {
				int num_operations = (1 << (d + 1));
				blockNum = (pow2len / num_operations + blockSize) / blockSize;
				cudaSweepDown<<<blockNum, blockSize >>>(pow2len / num_operations, d, dev_data);
			}
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			checkCUDAError("cudaMmcpy dev_data to odata failed!");

			cudaFree(dev_data);
			checkCUDAError("cudaFree dev_data failed!");

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
            //timer().startGpuTimer();
            // TODO

			int *dev_idata, *dev_odata, *dev_bool_data;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_bool_data, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_bool_data failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			//bool Mapping
			blockNum = (n + blockSize) / blockSize;

			Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, dev_bool_data, dev_idata);

			int *bool_data = new int[n];
			//Save bool array to bool_data
			cudaMemcpy(bool_data, dev_bool_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMalloc dev_bool_data to bool_data failed!");

			int *indices = new int[n];
			//Compute indices array

			//call scan function
			scan(n, indices, bool_data);
			int count = indices[n - 1] + bool_data[n - 1];

			int *dev_indices;
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");

			cudaMemcpy(dev_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMalloc indices to dev_indices failed!");
			
			//Scattered
			blockNum = (n + blockSize) / blockSize;

			Common::kernScatter<<<blockNum, blockSize>>>(n, dev_odata, dev_idata, dev_bool_data, dev_indices);
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMalloc dev_odata to odata failed!");

			//Free data
			cudaFree(dev_idata);
			checkCUDAError("cudaFree dev_idata failed!");
			cudaFree(dev_odata);
			checkCUDAError("cudaFree dev_idata failed!");
			cudaFree(dev_bool_data);
			checkCUDAError("cudaFree dev_idata failed!");
			cudaFree(dev_indices);
			checkCUDAError("cudaFree dev_idata failed!");

			delete indices;
			delete bool_data;

			//timer().endGpuTimer();
            return count;
        }
    }
}
