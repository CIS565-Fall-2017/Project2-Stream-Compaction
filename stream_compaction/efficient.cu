#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

		#define blockSize 128

		__global__ void kernUpSweep(int N, int stride, int halfStride, int * data)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			
			if (index >= N)
				return;

			index = (index + 1) * stride - 1;
			data[index] += data[index - halfStride];
		}

		__global__ void kernDownSweepFirst(int N, int stride, int halfStride, int * data)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index >= N)
				return;

			index = (index + 1) * stride - 1;
			int tmp = data[index - halfStride];

			// Swap
			data[index - halfStride] = 0;

			// Add, replace
			data[index] = tmp;
		}

		__global__ void kernDownSweep(int N, int stride, int halfStride, int * data)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			
			if (index >= N)
				return;

			index = (index + 1) * stride - 1;

			int value = data[index];
			int tmp = data[index - halfStride];

			// Swap
			data[index - halfStride] = value;

			// Add, replace
			data[index] = value + tmp;
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		
		void scan(int n, int *dev_data)
		{
			int passes = ilog2ceil(n);
			for (int d = 0; d < passes; ++d)
			{
				int stride = pow(2, d + 1);
				int halfStride = stride / 2;
				int sliceElements = n / stride;

				//printf("%d, %d, %d \n", sliceElements, stride, halfStride);
				dim3 blocks((sliceElements + blockSize - 1) / blockSize);

				kernUpSweep << <blocks, blockSize >> > (sliceElements, stride, halfStride, dev_data);
				checkCUDAErrorFn("kernUpSweep failed!");
			}

			for (int d = passes - 1; d >= 0; --d)
			{
				int stride = pow(2, d + 1);
				int halfStride = stride / 2;
				int sliceElements = n / stride;

				//printf("%d, %d, %d \n", sliceElements, stride, halfStride);
				dim3 blocks((sliceElements + blockSize - 1) / blockSize);

				if (d == passes - 1)
				{
					kernDownSweepFirst << <blocks, blockSize >> > (sliceElements, stride, halfStride, dev_data);
					checkCUDAErrorFn("kernDownSweepFirst failed!");
				}
				else
				{
					kernDownSweep << <blocks, blockSize >> > (sliceElements, stride, halfStride, dev_data);
					checkCUDAErrorFn("kernDownSweep failed!");
				}
			}
        }

		void scan(int n, int *odata, const int *idata)
		{
			int * dev_data;
			int passes = ilog2ceil(n);
			int squareN = pow(2, passes);

			//printf("%d vs %d\n", n, squareN);

			cudaMalloc((void**)&dev_data, squareN * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_data failed!");

			// calloc
			cudaMemset(dev_data, 0, squareN * sizeof(int));
			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy dev_data failed!");

			timer().startGpuTimer();

			scan(squareN, dev_data);

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy dev_data failed!");
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
        int compact(int n, int *odata, const int *idata) 
		{
			int * dev_data;
			int * dev_booleans;
			int * dev_data_output;
			int passes = ilog2ceil(n);
			int squareN = pow(2, passes);

			cudaMalloc((void**)&dev_data_output, squareN * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_data failed!");

			cudaMalloc((void**)&dev_booleans, squareN * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_data failed!");

			cudaMalloc((void**)&dev_data, squareN * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_data failed!");

			// calloc
			cudaMemset(dev_data_output, 0, squareN * sizeof(int));
			cudaMemset(dev_data, 0, squareN * sizeof(int));
			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy dev_data failed!");

            timer().startGpuTimer();

			dim3 blocks((squareN + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <blocks, blockSize >> > (squareN, dev_booleans, dev_data);
			checkCUDAErrorFn("kernMapToBoolean failed!");

			scan(squareN, dev_booleans);

			int sum = 0;
			cudaMemcpy(&sum, &dev_booleans[squareN-1], sizeof(int), cudaMemcpyDeviceToHost);
			
			// Note: I removed one of the input arrays
			StreamCompaction::Common::kernScatter << <blocks, blockSize >> > (squareN, dev_data_output, dev_data, dev_booleans);
			checkCUDAErrorFn("kernScatter failed!");

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_data_output, sizeof(int) * sum, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpy dev_booleans failed!");

			cudaFree(dev_data);
			cudaFree(dev_data_output);
			cudaFree(dev_booleans);

			return sum;
        }
    }
}
