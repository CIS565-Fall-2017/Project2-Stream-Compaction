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
		
		void printArray(int n, int *a, bool abridged = false) {
			printf("    [ ");
			for (int i = 0; i < n; i++) {
				if (abridged && i + 2 == 5 && n > 10) {
					i = n - 5;
					printf("... ");
				}
				printf("%3d ", a[i]);
			}
			printf("]\n");
		}

		__global__ void kernelUpsweep(int *data, int n, int d) {
			int stride = 1 << d;
			//step to every (2*stride) indicies
			int index = ((blockIdx.x * blockDim.x) + threadIdx.x + 1) * 2 * stride;
			//sum accross the stride
			if (index < n+1) {
				data[index-1] += data[index-stride-1];
			}
		}

		__global__ void kernelDownsweep(int *data, int n, int d) {
			int stride = 1 << d;
			//step accross every (2*stride) indicies
			int index = ((blockIdx.x * blockDim.x) + threadIdx.x + 1) * 2 * stride;
			//switch right to left, add left to right
			if (index < n+1) {
				int temp = data[index - stride - 1];
				data[index - stride - 1] = data[index - 1];
				data[index - 1] += temp;
			}
		}

		__global__ void kernelBinary(int* odata, int* idata, int n) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				odata[index] = idata[index] ? 1 : 0;
			}
		}

		__global__ void kernelScatter(int *binary, int *scan, int *idata, int *odata, int n) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n && binary[index] != 0) {
				odata[scan[index]] = idata[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        
		void scan(int n, int *odata, const int *idata, bool isDevice) {
			//device array will be as long as next power of 2
			int maxStride = ilog2ceil(n);
			int POT_size = 1 << maxStride;

			//allocate array to length POT_size, and copy the first n values of odata
			int *dev_odata;
			if (isDevice) {
				cudaMalloc((void**)&dev_odata, POT_size * sizeof(int));
				cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			else {
				cudaMalloc((void**)&dev_odata, POT_size * sizeof(int));
				cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				timer().startGpuTimer();
			}
			cudaMemset(&dev_odata[n], 0, (POT_size-n) * sizeof(int));

			//determine resource needs
			dim3 fullBlocksPerGrid(1);
			dim3 threadsPerBlock(blockSize);
			
			//upsweep
			for (int i = 0; i < maxStride; i++) {
				fullBlocksPerGrid = dim3((POT_size / (1 << i) + blockSize - 1) / blockSize);
				kernelUpsweep <<<fullBlocksPerGrid, blockSize>>> (dev_odata, POT_size, i);
			}

			//set odata[POT_size-1] to 0
			int just_a_zero = 0;
			cudaMemcpy(&dev_odata[POT_size-1], &just_a_zero, sizeof(int), cudaMemcpyHostToDevice);

			//downsweep
			for (int i = maxStride; i >= 0; i--) {
				fullBlocksPerGrid = dim3((POT_size / (1 << i) + blockSize - 1) / blockSize);
				kernelDownsweep <<<fullBlocksPerGrid, blockSize>>> (dev_odata, POT_size, i);
			}

			//send data back to cpu memory
			if (!isDevice) {
				timer().endGpuTimer();
				cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
				cudaFree(dev_odata);
			}
			else {
				cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}
        }

		void scan(int n, int *odata, const int *idata) {
			scan(n, odata, idata, false);
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
            
			int *dev_scan, *dev_binary, *dev_odata, *dev_idata;
			int lastBinary, lastScan;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_scan, n * sizeof(int));
			cudaMalloc((void**)&dev_binary, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			

			dim3 blocksToMakeN((n + blockSize - 1) / blockSize);
			
			timer().startGpuTimer();
            //create binary array
			kernelBinary <<<blocksToMakeN, blockSize>>> (dev_binary, dev_idata, n);

			//scan said array
			scan(n, dev_scan, dev_binary, true);

			//scatter
			kernelScatter << <blocksToMakeN, blockSize >> > (dev_binary, dev_scan, dev_idata, dev_odata, n);

			//our odata buffer should now reflect the compacted stream

            timer().endGpuTimer();

			cudaMemcpy(&lastBinary, dev_binary + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&lastScan, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(odata, dev_odata, (lastScan + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_binary);
			cudaFree(dev_scan);
			cudaFree(dev_odata);
			cudaFree(dev_idata);

			return lastBinary ? lastScan + 1 : lastScan;
        }
    }
}
