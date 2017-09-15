#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernZeroed(int totalN, int n, int *odata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= totalN) return;

			odata[index] = (index < n) ? odata[index] : 0;
		}

		__global__ void kernUpSweep(int n, int d, int *odata) 
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			int offset = 1 << (d + 1);
			int i = (index + 1) * offset - 1;

			int val = 1 << d;
			odata[i] += odata[i - val];
			if (i == n - 1) odata[i] = 0;
		}

		__global__ void kernDownSweep(int n, int d, int *odata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			int i = index + 1;
			int val = 1 << d;
			int offset = 1 << (d + 1);
			int temp = odata[i * offset - 1];
			odata[i * offset - 1] += odata[i * offset - val - 1];
			odata[i * offset - val - 1] = temp;
		}

		void scan_implementation(int n, int *dev_out)
		{
			int pow2n = 1 << ilog2ceil(n);
			
			for (int d = 0; d < ilog2ceil(pow2n); ++d) {
				dim3 blocksPerGrid((pow2n / (1 << (d + 1)) + blockSize - 1) / blockSize);
				kernUpSweep << <blocksPerGrid, blockSize >> > (pow2n, d, dev_out);
				checkCUDAError("kernUpSweep failed!");
			}

			for (int d = ilog2ceil(pow2n) - 1; d >= 0; --d) {
				dim3 blocksPerGrid((pow2n / (1 << (d + 1)) + blockSize - 1) / blockSize);
				kernDownSweep << <blocksPerGrid, blockSize >> > (pow2n, d, dev_out);
				checkCUDAError("kernDownSweep failed!");
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
		{
			int *dev_out;
			int pow2n = 1 << ilog2ceil(n);

			cudaMalloc((void**)&dev_out, pow2n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			cudaMemcpy(dev_out, idata, pow2n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_out failed!");

			dim3 blocksPerGrid((pow2n + blockSize - 1) / blockSize);
			kernZeroed << <blocksPerGrid, blockSize >> > (pow2n, n, dev_out);
			checkCUDAError("kernZeroed failed!");
			
			timer().startGpuTimer();
			scan_implementation(n, dev_out);
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			cudaFree(dev_out);
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
			// TODO
			int *dbools;
			int *dev_in;
			int *dev_out;
			int *indices;

			cudaMalloc((void**)&dbools, (n + 1) * sizeof(int));
			checkCUDAError("cudaMalloc bools failed!");

			cudaMalloc((void**)&dev_in, (n + 1) * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");

			cudaMemcpy(dev_in, idata, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in failed!");

			timer().startGpuTimer();

			dim3 blocksPerGrid1((n + blockSize) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid1, blockSize >> > (n+1, dbools, dev_in);
			checkCUDAError("kernMapToBoolean failed!");

			scan_implementation(n + 1, dbools);

			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernScatter << <blocksPerGrid, blockSize >> > (n, dev_out, dev_in, dbools);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");

			int *num = (int *)malloc(sizeof(int));
			cudaMemcpy(num, dbools + n , sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpyDeviceToHost failed!");
            
            return *num;
        }
    }
}
