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
        //timer().startGpuTimer();

        int maxN = (1 << ilog2ceil(n));
        dim3 fullBlocksPerGrid((maxN + blockSize - 1) / blockSize);

        int* idata_swap;

        cudaMalloc((void**)&idata_swap, maxN * sizeof(int));
        checkCUDAError("cudaMalloc for idata_swap failed");

        cudaMemset(idata_swap, 0, maxN * sizeof(int));
        checkCUDAError("cudaMemset for idata_swap failed");

        // Copy from CPU to GPU
        cudaMemcpy(idata_swap, idata, n * sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy for idata_swap failed");

        // Up-sweep
        for (int depth = 0; depth < ilog2ceil(n); depth++) {
          int shift = (1 << depth);

          kernUpSweep << <fullBlocksPerGrid, blockSize >> >(maxN, idata_swap, shift);
          checkCUDAError("kernUpSweep failed");
        }

        cudaMemset(idata_swap + maxN - 1, 0, sizeof(int));
        
        // Down-sweep
        for (int depth = ilog2ceil(n) - 1; depth >= 0; depth--) {
          int shift = (1 << depth);

          kernDownSweep << <fullBlocksPerGrid, blockSize >> >(maxN, idata_swap, shift);
          checkCUDAError("kernUpSweep failed");
        }

        // Copy from GPU back to CPU
        cudaMemcpy(odata, idata_swap, n * sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy for idata_swap failed");

        cudaFree(idata_swap);
        
        //timer().endGpuTimer();
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
        int remainingN = 0;

        // Allocate extra buffers
        int* odata_swap;
        cudaMalloc((void**)&odata_swap, n * sizeof(int));
        checkCUDAError("cudaMalloc for odata_swap failed");

        int* idata_swap;
        cudaMalloc((void**)&idata_swap, n * sizeof(int));
        checkCUDAError("cudaMalloc for idata_swap failed");

        int* bools_arr;
        cudaMalloc((void**)&bools_arr, n * sizeof(int));
        checkCUDAError("cudaMalloc for temp_data failed");

        int* indices_arr;
        cudaMalloc((void**)&indices_arr, n * sizeof(int));
        checkCUDAError("cudaMalloc for scan_result failed");

        // Copy from CPU to GPU
        cudaMemcpy(odata_swap, odata, n * sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy for odata_swap failed");

        cudaMemcpy(idata_swap, idata, n * sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy for idata_swap failed");

        // Map input array to a temp array of 0s and 1s
        StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, bools_arr, idata_swap);
        checkCUDAError("kernMapToBoolean failed");

        // Scan
        scan(n, indices_arr, idata_swap);

        // Scatter
        StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, odata_swap, idata_swap, bools_arr, indices_arr);
        checkCUDAError("kernScatter failed");

        // Copy over compacted data from GPU to CPU
        cudaMemcpy(odata, odata_swap, n * sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy for odata_swap failed");

        cudaFree(odata_swap);
        cudaFree(idata_swap);
        cudaFree(bools_arr);
        cudaFree(indices_arr);
        
        timer().endGpuTimer();
        return remainingN;
    }
  }
}
