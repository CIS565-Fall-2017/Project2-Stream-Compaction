#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "sharedandbank.h"
namespace StreamCompaction {
	namespace SharedAndBank {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}

		//////would like to maximize the shared memory but we are thread limited(2048/mp)(32 blocks/mp) so block size 64
		////int determineBlockSizeForScan(const int totalBytes) {
		////	cudaDeviceProp deviceProp;
		////	int gpuDevice;
		////	cudaGetDeviceProperties(&deviceProp, gpuDevice);
		////	int block_shmem = deviceProp.sharedMemPerBlock;
		////	int block_thd = deviceProp.maxThreadsPerBlock;
		////	int mp_shmem = deviceProp.sharedMemPerMultiprocessor;
		////	int mp_thd = deviceProp.maxThreadsPerMultiProcessor;
		////	int mp_count = deviceProp.multiProcessorCount;
		////	int dev_shmem = mp_count * mp_shmem;
		////}


		//__global__ void kernZeroExcessLeaves(const int pow2roundedsize, const int orig_size, int* data) {
		//	const int index = blockIdx.x * blockDim.x + threadIdx.x;
		//	if (index >= pow2roundedsize || index < orig_size) return;
		//	data[index] = 0;
		//}

		//__global__ void kernScanDown(const int pow2roundedsize, const int indexscaling, const int offset, int* data) {
		//	//shift orig index up by 1 (otherwise thread 0 wouldn't pick up the index modifications), scale it, then shift back down
		//	const int index = (indexscaling * (blockIdx.x * blockDim.x + threadIdx.x + 1)) - 1;
		//	if (index >= pow2roundedsize) return;
		//	int oldparent = data[index];
		//	data[index] += data[index - offset];
		//	data[index - offset] = oldparent;
		//}

		//__global__ void kernScan(const int n, const int sharedMemEntries, int* SUMS, int* data) {
		//	extern __shared__ int temp[];
		//	const int thid_blk = threadIdx.x;
		//	const int thid_grid = blockIdx.x * blockDim.x + threadIdx.x;
		//	if (2*thid_grid+1 >= n) return;

		//	temp[2*thid_blk]   = data[2*thid_grid];
		//	temp[2*thid_blk+1] = data[2*thid_grid+1];

		//	//Scan upsweep
		//	int offset = 1;
		//	for (int d = sharedMemEntries >> 1; d > 0; d >>= 1) {
		//		__syncthreads();
		//		if (thid_blk < d) {
		//			int lchild = offset*(2 * thid_blk + 1) - 1;
		//			int rchild = offset*(2 * thid_blk + 2) - 1;
		//			temp[rchild] += temp[lchild];
		//		}
		//		offset <<= 1;
		//	}

		//	//intermediate step, copy the block sums to SUMS and scan that
		//	__syncthreads();
		//	if (gridDim.x > 1 && thid_blk == 0) { SUMS[blockIdx.x] = temp[sharedMemEntries - 1]; }
		//	//do a normal global memory up and sweep using thid_grid

		//	//add SUMS values from indices 0 to n-1 to all entries in blocks with blockIdx.x == 0 to gridDim.x-1

		//	//set last index
		//	if(thid_grid == gridDim.x*blockDim.x-1) {}

		//	//shared mem blockwise scan down requires you to see how many
		//}

		//void scan(const int n, int *odata, const int *idata) {
		//	int* dev_data;
		//	const int pow2roundedsize = 1 << ilog2ceil(n);
		//	const int numbytes_pow2roundedsize = pow2roundedsize * sizeof(int);
		//	const int numbytes_copy = n * sizeof(int);

		//	cudaMalloc((void**)&dev_data, numbytes_pow2roundedsize);
		//	checkCUDAError("cudaMalloc dev_data failed!");

		//	cudaMemcpy(dev_data, idata, numbytes_copy, cudaMemcpyHostToDevice);
		//	checkCUDAError("cudaMemcpy from idata to dev_data failed!");

		//	int gridDim = (pow2roundedsize + blockSize - 1) / blockSize;

		//	//the algo works on pow2 sized arrays so we size up the array to the next pow 2 if it wasn't a pow of 2 to begin with
		//	//then we need to fill data after index n-1 with zeros 
		//	kernZeroExcessLeaves<<<gridDim, blockSize>>>(pow2roundedsize, n, dev_data);

		//	//if (usetimer) { timer().startGpuTimer(); }
		//	timer().startGpuTimer();
		//	const int sharedMemEntries = 2 * blockSize;
		//	gridDim = ((pow2roundedsize >> 1) + blockSize - 1) / blockSize;
		//	kernScan << <gridDim, blockSize, sharedMemEntries * sizeof(int) >> > (pow2roundedsize, sharedMemEntries, dev_data);

		//	//make sure last index value is 0 before we downsweep
		//	const int zero = 0;
		//	cudaMemcpy(dev_data + pow2roundedsize - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
		//	checkCUDAError("cudaMemcpy from zero to dev_data failed!");

		//	for (int offset = pow2roundedsize >> 1; offset > 0; offset >>= 1) {
		//		gridDim = ((pow2roundedsize >> ilog2(offset<<)) + blockSize - 1) / blockSize;
		//		kernScanDown<<<gridDim, blockSize>>>(pow2roundedsize, offset << 1, offset, dev_data);
		//	}
		//	timer().endGpuTimer();

		//	cudaMemcpy(odata, dev_data, numbytes_copy, cudaMemcpyDeviceToHost);
		//	checkCUDAError("cudaMemcpy from dev_data to odata failed!");

		//	cudaFree(dev_data);
		//	checkCUDAError("cudaFree(dev_data) failed!");

		//}

		//int compact(const int n, int *odata, const int *idata) {
		//	return -1;
		//}
	}
}