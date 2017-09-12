#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
  namespace Naive {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }
    // TODO: __global__
    __global__ void kernScan(int n, int* odata, int* idata, int shift) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      if (index >= shift) {
        odata[index] = idata[index] + idata[index - shift];
      }
      else {
        odata[index] = idata[index];
      }
    }

    __global__ void kernExclusiveShift(int n, int* odata, int* idata) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      if (index > 0) {
        odata[index] = idata[index - 1];
      }
      else {
        odata[index] = 0;
      }
    }

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int *odata, const int *idata) {
      timer().startGpuTimer();
            
      dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

      int* odata_swap;
      int* idata_swap;
      cudaMalloc((void**)&odata_swap, n * sizeof(int));
      checkCUDAError("cudaMalloc for idata_swap failed");
      cudaMalloc((void**)&idata_swap, n * sizeof(int));
      checkCUDAError("cudaMalloc for odata_swap failed");

      // Copy from CPU to GPU
      cudaMemcpy(odata_swap, odata, n * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(idata_swap, idata, n * sizeof(int), cudaMemcpyHostToDevice);

      for (int depth = 1; depth <= ilog2ceil(n); depth++) {
        int shift = 1;
        if (depth > 1) {
          shift = 2 << (depth - 2);
        }

        kernScan << <fullBlocksPerGrid, blockSize >> >(n, odata_swap, idata_swap, shift);
        checkCUDAError("kernScan failed");

        // Swap buffers for next iteration
        cudaMemcpy(idata_swap, odata_swap, n * sizeof(int), cudaMemcpyDeviceToDevice);
      }

      kernExclusiveShift << <fullBlocksPerGrid, blockSize >> >(n, odata_swap, idata_swap);
      checkCUDAError("kernInclusiveShift failed");

      // Copy from GPU back to CPU
      cudaMemcpy(odata, odata_swap, n * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(odata_swap);
      cudaFree(idata_swap);

      timer().endGpuTimer();
    }
  }
}
