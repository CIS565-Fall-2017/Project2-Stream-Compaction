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

		__global__ void kernReduction(int n, int stepSize, int* data) {
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			// scale the index to account for thread number < array size
			idx++; // finding right element of pair for reduction
			// e.g. thread 0 adds data[3] and data[3 - 2 = 1]
			idx *= stepSize * 2;
			idx--;
			if (idx >= n) return;
			data[idx] += data[idx - stepSize];
		}

		__global__ void kernDownSweep(int n, int stepSize, int* data) {
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			// same idx as kernReduction, done in reverse order
			idx++;
			idx *= stepSize * 2;
			idx--;
			if (idx >= n) return;
			int temp = data[idx];
			data[idx] += data[idx - stepSize];
			data[idx - stepSize] = temp;
		}

		__global__ void kernPadZeros(int oldN, int newN, int* data) {
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= newN || idx < oldN) return;
			data[idx] = 0;
		}

		__global__ void kernZeroArray(int n, int* data) {
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= n) return;
			data[idx] = 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			
			int lvls = ilog2ceil(n);
			int paddedSize = 1 << lvls;
			int* dev_scan;

			cudaMalloc((void**)&dev_scan, paddedSize * sizeof(int));
			checkCUDAError("scan efficient allocation failed");

			cudaMemcpy(dev_scan, idata, paddedSize * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("scan memcpy failed");

			int blockSize = 128;
			dim3 arrCountSize((paddedSize + blockSize - 1) / blockSize);

			kernPadZeros << <arrCountSize, blockSize >> >(n, paddedSize, dev_scan);
			
			timer().startGpuTimer();

			int numThreads = paddedSize >> 1; // start w/ every other index
			for (int step = 1; step < paddedSize; step = step << 1) {
				arrCountSize = (numThreads + blockSize - 1) / blockSize;
				kernReduction << <arrCountSize, blockSize >> >(paddedSize, step, dev_scan);
				numThreads = numThreads >> 1;
			}

			// need 0 in final index
			const int temp = 0;
			cudaMemcpy(dev_scan + paddedSize - 1, &temp, sizeof(int), cudaMemcpyHostToDevice);

			numThreads = 1;
			for (int step = paddedSize >> 1; step > 0; step = step >> 1) {
				arrCountSize = (numThreads + blockSize - 1) / blockSize;
				kernDownSweep << <arrCountSize, blockSize >> >(paddedSize, step, dev_scan);
				numThreads = numThreads << 1;
			}

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_scan, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_scan);
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
			int* dev_bools;
			int* dev_indices;
			int* dev_idata;
			int* dev_odata;

			int lvls = ilog2ceil(n);
			int paddedSize = 1 << lvls;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("compact idata allocation failed");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("copying compact idata failed");

			cudaMalloc((void**)&dev_bools, paddedSize * sizeof(int));
			checkCUDAError("compact bools allocation failed");

			int blockSize = 128;
			dim3 padCountSize((paddedSize + blockSize - 1) / blockSize);
			dim3 nCountSize((n + blockSize - 1) / blockSize);
			kernZeroArray << <padCountSize, blockSize >> >(paddedSize, dev_bools);


			cudaMalloc((void**)&dev_indices, paddedSize * sizeof(int));
			checkCUDAError("compact bools allocation failed");

			// could prob. allocate with trimmed size, but want to allocate before timer
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("allocating out data failed");

			timer().startGpuTimer();
            // map the booleans
			StreamCompaction::Common::kernMapToBoolean << <nCountSize, blockSize >> >(n, dev_bools, dev_idata);

			// copy booleans to indices to scan in place
			cudaMemcpy(dev_indices, dev_bools, paddedSize * sizeof(int), cudaMemcpyDeviceToDevice);

			// scan
			int numThreads = paddedSize >> 1; // start w/ every other index
			for (int step = 1; step < paddedSize; step = step << 1) {
				padCountSize = (numThreads + blockSize - 1) / blockSize;
				kernReduction << <padCountSize, blockSize >> >(paddedSize, step, dev_indices);
				numThreads = numThreads >> 1;
			}

			// need 0 in final index
			const int temp = 0;
			cudaMemcpy(dev_indices + paddedSize - 1, &temp, sizeof(int), cudaMemcpyHostToDevice);

			numThreads = 1;
			for (int step = paddedSize >> 1; step > 0; step = step >> 1) {
				padCountSize = (numThreads + blockSize - 1) / blockSize;
				kernDownSweep << <padCountSize, blockSize >> >(paddedSize, step, dev_indices);
				numThreads = numThreads << 1;
			}

			// find the size of the out data
			int lastIndex;
			int lastBool;

			cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("getting last boolean failed");
			cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("getting last index failed");

			int outSize = lastIndex + lastBool;

			StreamCompaction::Common::kernScatter << <nCountSize, blockSize >> >(n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, outSize * sizeof(int), cudaMemcpyDeviceToDevice);

			cudaFree(dev_bools);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_indices);

            return outSize;
        }
    }
}
