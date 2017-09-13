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

    __global__ void kernUpSweep(int n, int* idata, int shift) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      int offset = (shift << 1);
      if (index % offset == 0 && index + offset <= n) {
        idata[index + offset - 1] += idata[index + shift - 1];
      }
    }

    __global__ void kernDownSweep(int n, int* idata, int shift) {
      int index = threadIdx.x + (blockIdx.x * blockDim.x);
      if (index >= n) {
        return;
      }

      int offset = (shift << 1);
      if (index % offset == 0 && index + offset <= n) {
        int temp = idata[index + shift - 1];
        idata[index + shift - 1] = idata[index + offset - 1];
        idata[index + offset - 1] += temp;
      }
    }

    /**
      * Performs prefix-sum (aka scan) on idata, storing the result into odata.
      */
    void scan(int n, int *odata, const int *idata) {
      timer().startGpuTimer();

      int maxN = (1 << ilog2ceil(n));
      dim3 fullBlocksPerGrid((maxN + blockSize - 1) / blockSize);

      int* idataSwap;

      cudaMalloc((void**)&idataSwap, maxN * sizeof(int));
      checkCUDAError("cudaMalloc for idata_swap failed");

      cudaMemset(idataSwap, 0, maxN * sizeof(int));
      checkCUDAError("cudaMemset for idata_swap failed");

      // Copy from CPU to GPU
      cudaMemcpy(idataSwap, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for idata_swap failed");

      // Up-sweep
      for (int depth = 0; depth < ilog2ceil(n); depth++) {
        int shift = (1 << depth);

        kernUpSweep << <fullBlocksPerGrid, blockSize >> >(maxN, idataSwap, shift);
        checkCUDAError("kernUpSweep failed");
      }

      cudaMemset(idataSwap + maxN - 1, 0, sizeof(int));
        
      // Down-sweep
      for (int depth = ilog2ceil(n) - 1; depth >= 0; depth--) {
        int shift = (1 << depth);

        kernDownSweep << <fullBlocksPerGrid, blockSize >> >(maxN, idataSwap, shift);
        checkCUDAError("kernUpSweep failed");
      }

      // Copy from GPU back to CPU
      cudaMemcpy(odata, idataSwap, n * sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy for idata_swap failed");

      cudaFree(idataSwap);
        
      timer().endGpuTimer();
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
        
      dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

      // Allocate extra buffers
      int* odataSwap;
      cudaMalloc((void**)&odataSwap, n * sizeof(int));
      checkCUDAError("cudaMalloc for odataSwap failed");

      int* idataSwap;
      cudaMalloc((void**)&idataSwap, n * sizeof(int));
      checkCUDAError("cudaMalloc for idataSwap failed");

      int* boolsArr;
      cudaMalloc((void**)&boolsArr, n * sizeof(int));
      checkCUDAError("cudaMalloc for boolsArr failed");

      int* indicesArr;
      cudaMalloc((void**)&indicesArr, n * sizeof(int));
      checkCUDAError("cudaMalloc for indicesArr failed");

      // Copy from CPU to GPU
      cudaMemcpy(odataSwap, odata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for odataSwap failed");

      cudaMemcpy(idataSwap, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy for idataSwap failed");

      // Map input array to a temp array of 0s and 1s
      StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, boolsArr, idataSwap);
      checkCUDAError("kernMapToBoolean failed");

      // Scan
      scan(n, indicesArr, boolsArr);

      // Scatter
      StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, odataSwap, idataSwap, boolsArr, indicesArr);
      checkCUDAError("kernScatter failed");

      // Copy over compacted data from GPU to CPU
      cudaMemcpy(odata, odataSwap, n * sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy for odataSwap failed");

		  // Grab remaining number of elements
		  int remainingNBools = 0;
		  cudaMemcpy(&remainingNBools, boolsArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

		  int remainingNIndices = 0;
		  cudaMemcpy(&remainingNIndices, indicesArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

      int result = remainingNBools + remainingNIndices;
	
      cudaFree(odataSwap);
      cudaFree(idataSwap);
      cudaFree(boolsArr);
      cudaFree(indicesArr);
        
      timer().endGpuTimer();
      return result;
    }
  }
}
