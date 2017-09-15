#include <cuda.h>
#include <cuda_runtime.h>
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

		__global__ void kernScanUp(const int pow2roundedsize, const int indexscaling, const int offset, int* data) {
			//shift orig index up by 1 (otherwise thread 0 wouldn't pick up the index modifications), scale it, then shift back down
			const int index = (indexscaling * (blockIdx.x * blockDim.x + threadIdx.x + 1)) - 1;
			if (index >= pow2roundedsize) return;
			data[index] += data[index - offset];
		}

		__global__ void kernScanDown(const int pow2roundedsize, const int indexscaling, const int offset, int* data) {
			//shift orig index up by 1 (otherwise thread 0 wouldn't pick up the index modifications), scale it, then shift back down
			const int index = (indexscaling * (blockIdx.x * blockDim.x + threadIdx.x + 1)) - 1;
			if (index >= pow2roundedsize) return;
			int oldparent = data[index];
			data[index] += data[index - offset];
			data[index - offset] = oldparent;
		}

		__global__ void kernZeroExcessLeaves(const int pow2roundedsize, const int orig_size, int* data) {
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= pow2roundedsize || index < orig_size) return;
			data[index] = 0;
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(const int n, int *odata, const int *idata) {
			int* dev_data;
			const int pow2roundedsize = 1 << ilog2ceil(n);
			const int numbytes_pow2roundedsize = pow2roundedsize * sizeof(int);
			const int numbytes_copy = n * sizeof(int);

			cudaMalloc((void**)&dev_data, numbytes_pow2roundedsize);
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_data, idata, numbytes_copy, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from idata to dev_idata failed!");

			const dim3 gridDims( (pow2roundedsize + blockSize - 1) / blockSize, 0, 0 );
			const dim3 blockDims(blockSize, 0, 0);

			//the algo works on pow2 sized arrays so we size up the array to the next pow 2 if it wasn't a pow of 2 to begin with
			//then we need to fill data after index n-1 with zeros 
			kernZeroExcessLeaves<<<gridDims, blockDims>>>(pow2roundedsize, n, dev_data);

            timer().startGpuTimer();
			for (int offset = 1; offset < pow2roundedsize; offset <<= 1) {
				//gridDims.x can probably = pow2roundedsize >> ilog2(offset);
				kernScanUp<<<gridDims, blockDims>>>(pow2roundedsize, offset << 1, offset, dev_data);
			}

			//make sure last index value is 0 before we downsweep
			cudaThreadSynchronize();
			int zero = 0;
			cudaMemcpy(dev_data + pow2roundedsize - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy from zero to dev_data failed!");

			for (int offset = pow2roundedsize >> 1; offset > 0; offset >>= 1) {
				//gridDims can probably = pow2roundedsize >> ilog2(offset);
				kernScanDown<<<gridDims, blockDims>>>(pow2roundedsize, offset << 1, offset, dev_data);
			}
            timer().endGpuTimer();
			cudaThreadSynchronize();

			cudaMemcpy(odata, dev_data, numbytes_copy, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from dev_data to odata failed!");

			cudaFree(dev_data);
			checkCUDAError("cudaFree(dev_data) failed!");
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
        int compact(const int n, int *odata, const int *idata) {
			int* dev_idata;
			int* dev_odata;
			int* dev_bools;
			int* bools = new int[n];
			int* indices = new int[n];

			const int numbytes_copy = n * sizeof(int);

			cudaMalloc((void**)&dev_idata, numbytes_copy);
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)&dev_odata, numbytes_copy);
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMalloc((void**)&dev_bools, numbytes_copy);
			checkCUDAError("cudaMalloc bools failed!");

			cudaMemcpy(dev_idata, idata, numbytes_copy, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");

			const dim3 gridDims( (n + blockSize - 1) / blockSize, 0, 0 );
			const dim3 blockDims(blockSize, 0, 0);

            timer().startGpuTimer();

			StreamCompaction::Common::kernMapToBoolean<<<gridDims, blockDims>>>(n, bools, dev_idata);

			cudaThreadSynchronize();
			cudaMemcpy(bools, dev_bools, numbytes_copy, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_bools to bools failed!");

			StreamCompaction::Efficient::scan(n, indices, bools);
			StreamCompaction::Common::kernScatter<<<gridDims,blockDims>>>(n, dev_odata, idata, bools, indices);

            timer().endGpuTimer();
			

			cudaThreadSynchronize();
			cudaMemcpy(odata, dev_odata, numbytes_copy, cudaMemcpyDeviceToHost);
			const int size = indices[n - 1] + bools[n - 1]; 

			delete[] bools;
			delete[] indices;

			cudaFree(dev_idata);
			checkCUDAError("cudaFree of dev_idata failed!");

			cudaFree(dev_odata);
			checkCUDAError("cudaFree of dev_odata failed!");
			
			cudaFree(dev_bools);
			checkCUDAError("cudaFree of dev_bools failed!");

			return size;
        }
    }
}
