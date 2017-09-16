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

        __global__ void kernScan(int n, const int pow, int *odata, const int *idata) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index >= n) return;

          odata[index] = (index >= pow) ? idata[index - pow] + idata[index] : idata[index];
        }

        __global__ void kernInToEx(int n, int *odata, const int *idata) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index >= n) return;

          odata[index] = (index == 0) ? 0 : idata[index - 1];
        }

        // Kernel to pad the new array with 0s
        __global__ void kernPadWithZeros(const int n, const int nPad, int *dev_data) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index >= nPad || index < n) return;

          dev_data[index] = 0;
        }

        // Up-Sweep Kernel
        __global__ void kernUpSweep(const int n, const int pow, const int pow1, int *dev_data) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index % pow1 != pow1 - 1) return;
          dev_data[index] += dev_data[index - pow];
        }

        // Down-Sweep Kernel
        __global__ void kernDownSweep(const int n, const int pow, const int pow1, int *dev_data) {
          int index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index % pow1 != pow1 - 1) return;
          int t = dev_data[index - pow];
          dev_data[index - pow] = dev_data[index];
          dev_data[index] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          int nSize = n * sizeof(int);
          int nl = ilog2ceil(n);
          int nPad = 1 << nl;
          int nPadSize = nPad * sizeof(int);

          // Compute blocks per grid and threads per block
          dim3 numBlocks((nPad + blockSize - 1) / blockSize);
          dim3 numThreads(blockSize);

          int *dev_data;
          cudaMalloc((void**)&dev_data, nPadSize);
          checkCUDAError("cudaMalloc for dev_data failed!");

          // Copy device arrays to device
          cudaMemcpy(dev_data, idata, nSize, cudaMemcpyHostToDevice); // use a kernel to fill 0s for the remaining indices..
          checkCUDAError("cudaMemcpy for dev_data failed!");
          
          // Fill the padded part of dev_data with 0s..
          kernPadWithZeros <<<numBlocks, numThreads >>> (n, nPad, dev_data);

          timer().startGpuTimer();
          // Work Efficient Scan - Creates exclusive scan output

          for (int d = 0; d < nl; d++) {
            int pow = 1 << (d);
            int pow1 = 1 << (d + 1);
            kernUpSweep <<<numBlocks, numThreads>>> (nPad, pow, pow1, dev_data);
            checkCUDAError("kernUpSweep failed!");
          }
          
          //dev_data[nPad - 1] = 0; // set last element to 0 before downsweep.. 
          int zero = 0;
          cudaMemcpy(dev_data + nPad - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

          for (int d = nl - 1; d >= 0; d--) {
            int pow = 1 << (d);
            int pow1 = 1 << (d + 1);
            kernDownSweep <<<numBlocks, numThreads>>> (nPad, pow, pow1, dev_data);
            checkCUDAError("kernDownSweep failed!");
          }

          timer().endGpuTimer();

          // Copy device arrays back to host
          cudaMemcpy(odata, dev_data, nSize, cudaMemcpyDeviceToHost);
          checkCUDAError("cudaMemcpy (device to host) for odata failed!");

          // Free memory
          cudaFree(dev_data);
          checkCUDAError("cudaFree failed!");
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
