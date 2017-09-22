#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "sharedAndIndexing.h"

namespace StreamCompaction {
	namespace SharedAndIndexing {

#define blockSize 1024

		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernScan_UpSweep(int N_by_PowPlusOne, int* scan_out, int powerPlusOne, int power)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N_by_PowPlusOne)
			{
				return;
			}

			index = (index + 1)*powerPlusOne-1;
			scan_out[index] += scan_out[index - power];
		}

		__global__ void kernScan_DownSweep(int N_by_PowPlusOne, int* scan_out, int powerPlusOne, int power)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N_by_PowPlusOne)
			{
				return;
			}

			index = (index + 1)*powerPlusOne - 1;

			int temp = scan_out[index - power];
			scan_out[index - power] = scan_out[index];
			scan_out[index] += temp;
		}

		__global__ void kernExcessZeroFill(int pow2RoundedSize, int originalSize, int* scan_out)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x + originalSize +1;

			if (index < pow2RoundedSize)
			{
				scan_out[index] = 0;
			}
		}

		void scanImplementation(int log_n_ceil, int n, int pow2RoundedSize, int* dev_scan_out)
		{
			for (int i = 0; i <= log_n_ceil - 1; i++)
			{
				int two_power_d = 1 << i;
				int two_power_d_plus_one = two_power_d << 1;
				int N_by_PowPlusOne = pow2RoundedSize /two_power_d_plus_one;
					
				dim3 fullBlocksPerGrid_Strided(((pow2RoundedSize/two_power_d_plus_one) + blockSize - 1) / blockSize);
				kernScan_UpSweep <<<fullBlocksPerGrid_Strided, blockSize>>> (N_by_PowPlusOne, dev_scan_out, 
																			 two_power_d_plus_one, two_power_d);
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
				int N_by_PowPlusOne = pow2RoundedSize / two_power_d_plus_one;

				dim3 fullBlocksPerGrid_Strided(((pow2RoundedSize / two_power_d_plus_one) + blockSize - 1) / blockSize);
				kernScan_DownSweep <<<fullBlocksPerGrid_Strided, blockSize >>> (N_by_PowPlusOne, dev_scan_out,
																				two_power_d_plus_one, two_power_d);
				checkCUDAError("DownSweep Failed!");
			}
		}

		__global__ void kernScan_SharedMemory(int N, int* scan_out, int* sum)
		{
			extern __shared__ int shMemScanData[];
			const int threadID_block = threadIdx.x;
			const int threadID_grid = blockIdx.x * blockDim.x + threadIdx.x; 
			int offset = 1;

			// load input(which exists in dev_scan because work efficient scan happens in place) into shared memory
			shMemScanData[2 * threadID_block] = scan_out[2 * threadID_block]; 
			shMemScanData[2 * threadID_block + 1] = scan_out[2 * threadID_block + 1];

			//UpSweep
			for (int d = N >> 1; d > 0; d >>= 1)
			{
				__syncthreads();

				if (threadID_block < d)
				{
					int leftChild = offset*(2 * threadID_block + 1) - 1;
					int rightChild = offset*(2 * threadID_block + 2) - 1;

					shMemScanData[rightChild] += shMemScanData[leftChild];
				}
				offset *= 2;
			}
			
			if (gridDim.x > 1 && threadID_block == 0) ///CHECK TODO
			{ 
				sums[blockIdx.x] = shMemScanData[N-1];
			}

			// Clear the last element
			if (threadID_block == 0) 
			{
				shMemScanData[N - 1] = 0;
			}

			//DownSweep
			for (int d = 1; d < N; d <<= 1)
			{
				offset >>= 1;
				__syncthreads();

				if (threadID_block < d)
				{
					int leftChild = offset*(2 * threadID_block + 1) - 1;
					int rightChild = offset*(2 * threadID_block + 2) - 1;

					float inPlaceParent = shMemScanData[leftChild];
					shMemScanData[leftChild] = shMemScanData[rightChild];
					shMemScanData[rightChild] += inPlaceParent;
				}
			}

			// Write Results to device memory
			scan_out[2 * threadID_grid] = shMemScanData[2 * threadID_block];
			scan_out[2 * threadID_grid + 1] = shMemScanData[2 * threadID_block + 1];
		}

		void scanImplementationWithSharedMemory(int log_n_ceil, int n, int pow2RoundedSize, int* dev_scan_out, int* dev_sum)
		{
			const int sharedMemorySize = pow2RoundedSize;
			dim3 fullBlocksPerGrid_Strided((pow2RoundedSize + blockSize - 1) / blockSize);

			kernScan_SharedMemory <<<fullBlocksPerGrid_Strided, blockSize, sharedMemorySize>>> (N, dev_scan_out, dev_sum);
			checkCUDAError("DownSweep Failed!");
		}
		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata)
		{
			int* dev_scan_out;
			int* dev_sum;

			const int log_n_ceil = ilog2ceil(n);
			const int pow2RoundedSize = 1 << log_n_ceil;
			const int numbytes_pow2roundedsize = pow2RoundedSize * sizeof(int);
			const int numbytes_ForCopying = n * sizeof(int);

			cudaMalloc((void**)&dev_scan_out, numbytes_pow2roundedsize);
			checkCUDAErrorFn("cudaMalloc dev_scan_out failed!");
			cudaMalloc((void**)&dev_sum, numbytes_pow2roundedsize);
			checkCUDAErrorFn("cudaMalloc dev_scan_out failed!");
			cudaMemset(dev_sum, 0, numbytes_pow2roundedsize);

			cudaMemcpy(dev_scan_out, idata, numbytes_ForCopying, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_scan_A failed!");

			dim3 fullBlocksPerGrid_Diff(((pow2RoundedSize-n) + blockSize - 1) / blockSize);
			//Fill up the array such that anything beyond the original size but less than the actual pow2roundedSize is zero
			kernExcessZeroFill <<<fullBlocksPerGrid_Diff, blockSize >>> (pow2RoundedSize, n, dev_scan_out);

			timer().startGpuTimer();
				//scanImplementation(log_n_ceil, n, pow2RoundedSize, dev_scan_out);

				scanImplementationWithSharedMemory(log_n_ceil, n, pow2RoundedSize, dev_scan_out, dev_sum);
			timer().endGpuTimer();

			cudaThreadSynchronize();

			cudaMemcpy(odata, dev_scan_out, numbytes_ForCopying, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_odata to odata failed!");

			cudaFree(dev_scan_out);
			checkCUDAErrorFn("cudaFree failed!");
		}
	}
}