#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {

		#define blockSize 1024

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernScan_UpSweep(int N, int* scan_out, int powerPlusOne, int power)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index >= N)
			{
				return;
			}

			if (index%powerPlusOne == 0) //to account for the jump by powerPlusOne in parallel
			{
				scan_out[index + powerPlusOne - 1] += scan_out[index + power - 1];
			}
		}

		__global__ void kernScan_DownSweep(int pow2RoundedSize, int originalSize, int* scan_out, int powerPlusOne, int power)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index >= pow2RoundedSize)
			{
				return;
			}

			if (index%powerPlusOne == 0) //to account for the jump by powerPlusOne in parallel
			{
				int temp = scan_out[index + power - 1];
				scan_out[index + power - 1] = scan_out[index + powerPlusOne - 1];
				scan_out[index + powerPlusOne - 1] += temp;
			}
		}

		__global__ void kernExcessZeroFill(int pow2RoundedSize, int originalSize, int* scan_out)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index > originalSize && index < pow2RoundedSize)
			{
				scan_out[index] = 0;
			}
		}

		void scanImplementation(int log_n_ceil, int n, int pow2RoundedSize, int* dev_scan_out, dim3 fullBlocksPerGrid)
		{
			for (int i = 0; i <= log_n_ceil - 1; i++)
			{
				int two_power_d = 1 << i;
				int two_power_d_plus_one = two_power_d << 1;

				kernScan_UpSweep <<<fullBlocksPerGrid, blockSize>>> (n, dev_scan_out, two_power_d_plus_one, two_power_d);
				checkCUDAError("UpSweep Failed!");
			}

			//Ensure that the last index value is 0 before we execute downSweep
			const int zero = 0;
			cudaMemcpy(dev_scan_out + pow2RoundedSize - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from zero to dev_scan_out failed!");

			for (int i = log_n_ceil - 1; i >= 0; i--)
			{
				int two_power_d = 1 << i;
				int two_power_d_plus_one = two_power_d << 1;

				kernScan_DownSweep <<<fullBlocksPerGrid, blockSize>>> (pow2RoundedSize, n,
																	   dev_scan_out,
																	   two_power_d_plus_one, two_power_d);
				checkCUDAError("DownSweep Failed!");
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
		{
			int* dev_scan_out;

			const int log_n_ceil = ilog2ceil(n);
			const int pow2RoundedSize = 1 << log_n_ceil;
			const int numbytes_pow2roundedsize = pow2RoundedSize * sizeof(int);
			const int numbytes_ForCopying = n * sizeof(int);

			cudaMalloc((void**)&dev_scan_out, numbytes_pow2roundedsize);
			checkCUDAErrorFn("cudaMalloc dev_scan_out failed!");

			cudaMemcpy(dev_scan_out, idata, numbytes_ForCopying, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_scan_A failed!");

			dim3 fullBlocksPerGrid((pow2RoundedSize + blockSize - 1) / blockSize);

			//Fill up the array such that anything beyond the original size but less than the actual pow2roundedSize is zero
			kernExcessZeroFill <<<fullBlocksPerGrid, blockSize>>> (pow2RoundedSize, n, dev_scan_out);

			timer().startGpuTimer();
				scanImplementation(log_n_ceil, n, pow2RoundedSize, dev_scan_out, fullBlocksPerGrid);
			timer().endGpuTimer();

			cudaThreadSynchronize();

			cudaMemcpy(odata, dev_scan_out, numbytes_ForCopying, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_odata to odata failed!");

			cudaFree(dev_scan_out);
			checkCUDAErrorFn("cudaFree failed!");
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
			int* dev_bools;
			int* dev_indices;
			int* dev_idata;
			int* dev_odata;

			const int log_n_ceil = ilog2ceil(n);
			const int pow2RoundedSize = 1 << log_n_ceil;
			const int numbytes_pow2roundedsize = pow2RoundedSize * sizeof(int);
			const int numbytes_ForCopying = n * sizeof(int);

			dim3 fullBlocksPerGrid((pow2RoundedSize + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_bools, numbytes_pow2roundedsize);
			checkCUDAErrorFn("cudaMalloc dev_bools failed!");

			cudaMalloc((void**)&dev_indices, numbytes_pow2roundedsize);
			checkCUDAErrorFn("cudaMalloc dev_odata failed!");

			cudaMalloc((void**)&dev_idata, numbytes_pow2roundedsize);
			checkCUDAErrorFn("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)&dev_odata, numbytes_pow2roundedsize);
			checkCUDAErrorFn("cudaMalloc dev_odata failed!");

			cudaMemcpy(dev_idata, idata, numbytes_ForCopying, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_scan_A failed!");

			//Fill up the array such that anything beyond the original size but less than the actual pow2roundedSize is zero
			kernExcessZeroFill <<<fullBlocksPerGrid, blockSize>>> (pow2RoundedSize, n, dev_idata);

            timer().startGpuTimer();            
				//Create bool array from idata
				StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize>>> (n, dev_bools, dev_idata);
				//Copy bool data into indices array
				cudaMemcpy(dev_indices, dev_bools, numbytes_pow2roundedsize, cudaMemcpyDeviceToDevice);
				checkCUDAError("cudaMemcpy dev_indices failed");
				//Run scan on indices array
				scanImplementation(log_n_ceil, n, pow2RoundedSize, dev_indices, fullBlocksPerGrid);
				//Run scatter
				StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize>>> (n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();

			cudaThreadSynchronize();

			int newSize = -1;
			int* temp = new int[2];
			cudaMemcpy(&temp[0], dev_indices + pow2RoundedSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from zero to dev_scan_out failed!");
			cudaMemcpy(&temp[1], dev_bools + pow2RoundedSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from zero to dev_scan_out failed!");

			newSize = temp[0] + temp[1];

			cudaMemcpy(odata, dev_odata, newSize * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_odata to odata failed!");

			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			checkCUDAErrorFn("cudaFree failed!");

			return newSize;
        }
    }
}
