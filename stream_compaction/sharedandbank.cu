#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "sharedandbank.h"

int** scannedSUMS;

namespace StreamCompaction {
	namespace SharedAndBank {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernZeroExcessLeaves(const int pow2roundedsize, const int orig_size, int* data) {
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= pow2roundedsize || index < orig_size) return;
			data[index] = 0;
		}

#define AVOIDBANKCONFLICT 1
		__global__ void kernScan(const int shMemEntries, int* odata, const int* idata, int* SUMS) {
			extern __shared__ int temp[];
			const int thid_blk = threadIdx.x;
			const int thid_grid = blockIdx.x * blockDim.x + threadIdx.x;
			

#if AVOIDBANKCONFLICT == 1
			const int ai = thid_blk;
			const int bi = thid_blk + blockDim.x;
			const int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
			const int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
			const int ai_grid = blockIdx.x*shMemEntries + thid_blk;
			const int bi_grid = ai_grid + blockDim.x;
			temp[ai + bankOffsetA] = idata[ai_grid];
			temp[bi + bankOffsetB] = idata[bi_grid];
#else
			temp[2*thid_blk]   = idata[2*thid_grid];
			temp[2*thid_blk+1] = idata[2*thid_grid+1];
#endif

			//Scan upswep
			int offset = 1;
			for (int d = shMemEntries>>1; d > 0; d >>= 1) {//runs ilog2(shMemEntries) number of times
				__syncthreads();
				if (thid_blk < d) {//last iter offset is 64, lchild should be 63 and rchild 127
					int lchild = offset*(2*thid_blk+1)-1;
					int rchild = offset*(2*thid_blk+2)-1;
#if AVOIDBANKCONFLICT == 1
					lchild += CONFLICT_FREE_OFFSET(lchild);
					rchild += CONFLICT_FREE_OFFSET(rchild);
#endif
					temp[rchild] += temp[lchild];
				}
				offset <<= 1;
			}

			//intermediate step, copy the block sums to SUMS 
			int lastindex = shMemEntries - 1;
#if AVOIDBANKCONFLICT == 1
			lastindex += CONFLICT_FREE_OFFSET(lastindex);
#endif

			if (gridDim.x > 1 && 0 == thid_blk) { SUMS[blockIdx.x] = temp[lastindex]; }

			//zero last element for this block
			if (0 == thid_blk) { temp[lastindex] = 0; }

			//scan downswep
			for (int d = 1; d < shMemEntries; d <<= 1) {//runs same amount as downsweep
				offset >>= 1;
				__syncthreads();
				if (thid_blk < d) {
					int lchild = offset*(2*thid_blk+1)-1;
					int rchild = offset*(2*thid_blk+2)-1;
#if AVOIDBANKCONFLICT == 1
					lchild += CONFLICT_FREE_OFFSET(lchild);
					rchild += CONFLICT_FREE_OFFSET(rchild);
#endif
					int otherparent = temp[lchild];
					temp[lchild] = temp[rchild];
					temp[rchild] += otherparent;
				}
			}
			__syncthreads();

#if AVOIDBANKCONFLICT == 1
			odata[ai_grid] = temp[ai + bankOffsetA];
			odata[bi_grid] = temp[bi + bankOffsetB];
#else
			odata[2*thid_grid]   = temp[2*thid_blk];
			odata[2*thid_grid+1] = temp[2*thid_blk+1];
#endif
		}

		__global__ void kernAddBack(const int n, int* odata, const int* scannedSumsLevel) {
			__shared__ int scannedSumForThisBlock;
			if (threadIdx.x == 0) { scannedSumForThisBlock = scannedSumsLevel[blockIdx.x]; }
			const int thid_grid = blockIdx.x*blockDim.x + threadIdx.x;
			__syncthreads();
			odata[2*thid_grid] += scannedSumForThisBlock;//add running total of all prev elements before this block to this block
			odata[2*thid_grid+1] += scannedSumForThisBlock;
		}

		void recursiveScan(const int n, const int level, int *odata, const int *idata) {
			//printf("\ncalling recursiveScan with pow2size: %i level: %i\n", n, level);

			//generate params for the kernel
			const int shMemEntries = blockSize << 1;
			const int shMemSize = shMemEntries * sizeof(int);
			const int blocksThisLevel = (n + shMemEntries - 1) / shMemEntries;
			const int pow2BlocksThisLevel = 1 << ilog2ceil(blocksThisLevel);


			//check if we are at the last level
			//via how many blocks have on this level
			//if not keep recursing 
			if (pow2BlocksThisLevel > 1) {

				//1. scan up and down this level, if its the first recursion
				//then the thread blocks just scan sections of 
				//their corresponding global input data.
				//2. recursiveScan on the last entries in each block
				//(they get copied to an array stored at scannedSUMS[level]
				//for this level during the kernel call
				//3. call the addBack kernel so we can recursively 
				//add back the reduced sums of each block back up through
				//scannedSUMS levels and then finally back to the final 
				//result odata. Doing this will allow us to arrive at our 
				//final inclusive scanned result. pretty cool. 
				kernScan<<<pow2BlocksThisLevel,blockSize,shMemSize>>>(
					shMemEntries, odata, idata, scannedSUMS[level]);

				recursiveScan(pow2BlocksThisLevel, level+1, scannedSUMS[level], scannedSUMS[level]);

				kernAddBack<<<pow2BlocksThisLevel,blockSize>>>(n, odata, scannedSUMS[level]);

			} else {
				//last level, just call the kernel, the recursive call that we
				//are currently in is for the last level of scannedSUMS[]
				//After this we start popping the recursive stack,
				//adding the SUMS back up the through the scannedSUMS levels
				//and then into the final result odata of the first recursive call

				kernScan<<<pow2BlocksThisLevel,blockSize,shMemSize>>>(
					shMemEntries, odata, idata, scannedSUMS[level]);
				//gpuErrchk(cudaPeekAtLastError());
				//gpuErrchk(cudaDeviceSynchronize());
			}
		}

		void scan(const int n, int *odata, const int *idata) {
			int* dev_idata;
			int* dev_odata;
			const int pow2roundedsize = 1 << ilog2ceil(n);
			const int numbytes_pow2roundedsize = pow2roundedsize * sizeof(int);
			const int numbytes_copy = n * sizeof(int);

			/////////////////////////////////////////////////////////
			//// ALLOC scannedSUMS[] NEEDED FOR RECURSION PASSES ////
			/////////////////////////////////////////////////////////
			//alloc SUMS device memory pointer array as this scan
			//process needs to be recursive for arbitrary array sizes and arbitrary block sizes
			const int shMemEntries = blockSize << 1;
			int scannedSUMSTotalLevels;
			//int* scannedSUMSEntriesPerLevel;
			{
				int level = 0;
				int blocksThisLevel = (pow2roundedsize + shMemEntries - 1) / shMemEntries;
				while (blocksThisLevel > 1) {
					level++;
					blocksThisLevel = (blocksThisLevel + shMemEntries - 1) / shMemEntries;
				}
				scannedSUMS = (int**)malloc(level * sizeof(int*));
				scannedSUMSTotalLevels = level;
				//scannedSUMSEntriesPerLevel = new int[level];

				level = 0;
				blocksThisLevel = (pow2roundedsize + shMemEntries - 1) / shMemEntries;
				while (blocksThisLevel > 1) {
					const int pow2BlocksThisLevel = 1 << ilog2ceil(blocksThisLevel);
					//scannedSUMSEntriesPerLevel[level] = pow2BlocksThisLevel;
					cudaMalloc((void**)&scannedSUMS[level++], pow2BlocksThisLevel * sizeof(int));
					checkCUDAError("cudaMalloc scannedSUMS[level++] failed!");
					blocksThisLevel = (blocksThisLevel + shMemEntries - 1) / shMemEntries;
				}
			}

			/////////////////////////////////////////
			//// ALLOC AND COPY TO DEVICE MEMORY ////
			/////////////////////////////////////////
			cudaMalloc((void**)&dev_idata, numbytes_pow2roundedsize);
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaMalloc((void**)&dev_odata, numbytes_pow2roundedsize);
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaMemcpy(dev_idata, idata, numbytes_copy, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_data failed!");
			cudaMemcpy(dev_odata, idata, numbytes_copy, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_data failed!");

			/////////////////////////////
			//// 0 PAD FOR POW2 SIZE ////
			/////////////////////////////
			int gridDim = (pow2roundedsize + blockSize - 1) / blockSize;
			kernZeroExcessLeaves<<<gridDim, blockSize>>>(pow2roundedsize, n, dev_idata);
			kernZeroExcessLeaves<<<gridDim, blockSize>>>(pow2roundedsize, n, dev_odata);

			////////////////////////
			//// RECURSIVE SCAN ////
			////////////////////////
			timer().startGpuTimer();
			recursiveScan(pow2roundedsize, 0, dev_odata, dev_idata);
			timer().endGpuTimer();

			///////////////////////
			//// COPY AND FREE ////
			///////////////////////
			cudaMemcpy(odata, dev_odata, numbytes_copy, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_odata to odata failed!");
			cudaFree(dev_odata);
			checkCUDAError("cudaFree(dev_odata) failed!");
			cudaFree(dev_idata);
			checkCUDAError("cudaFree(dev_idata) failed!");
			{//free scannedSUMS related memory
				for (int i = 0; i < scannedSUMSTotalLevels; ++i) {
					cudaFree(scannedSUMS[i]);
					checkCUDAError("cudaFree(scannedSUMS[i]) failed!");
				}
				free(scannedSUMS);
			}
		}

		int compact(const int n, int *odata, const int *idata) {
			return -1;
		}
	}
}