#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        using namespace Common;
        bool compactTest = false;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int *g_odata, int n, int offset, int offsetPlus1) {

          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) return;
          if (index % offsetPlus1 == 0)
              g_odata[index + offsetPlus1 - 1] += g_odata[index + offset - 1];
        }

        __global__ void kernDownSweep(int *g_odata, int n, int offset, int offsetPlus1) {

          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) return;

          if (index % offsetPlus1 == 0) {
            int t = g_odata[index + offset - 1];
            g_odata[index + offset - 1] = g_odata[index + offsetPlus1 - 1];
            g_odata[index + offsetPlus1 - 1] += t;
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          int arrayLength = 1 << ilog2ceil(n);
          dim3 blocksPerGrid((arrayLength + BLOCK_SIZE - 1) / BLOCK_SIZE);
          int* h_idata = new int[arrayLength];
          memset(h_idata, 0, arrayLength * sizeof(int));
          memcpy(h_idata, idata, n * sizeof(int));
          int *d_idata;
          cudaMalloc((void**)&d_idata, arrayLength * sizeof(int));
          cudaMemcpy(d_idata, h_idata, arrayLength * sizeof(int), cudaMemcpyHostToDevice);

          if (!compactTest) timer().startGpuTimer();
          
          for (int d = 0; d <= ilog2ceil(n) - 1; d++)
            kernUpSweep << <blocksPerGrid, BLOCK_SIZE >> > (d_idata, arrayLength, 1 << d, 1 << d + 1);
          int zero = 0;
          cudaMemcpy(&d_idata[arrayLength-1], &zero, sizeof(int), cudaMemcpyHostToDevice);
          for (int d = ilog2ceil(n) - 1; d >= 0; d--)
            kernDownSweep << <blocksPerGrid, BLOCK_SIZE >> > (d_idata, arrayLength, 1 << d, 1 << d + 1);

          if (!compactTest) timer().endGpuTimer();

          cudaMemcpy(odata, d_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
          cudaFree(d_idata);
          delete[] h_idata;
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
          dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
          int* d_bools, *d_idata, *d_indicies, *d_odata;
          int* h_indicies = new int[n];
          int* h_bools = new int[n];

          cudaMalloc((void**)&d_bools, n * sizeof(int));
          cudaMalloc((void**)&d_idata, n * sizeof(int));
          cudaMalloc((void**)&d_indicies, n * sizeof(int));
          cudaMalloc((void**)&d_odata, n * sizeof(int));
          cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          compactTest = true;
          timer().startGpuTimer();
            
          kernMapToBoolean<<<blocksPerGrid, BLOCK_SIZE>>>(n, d_bools, d_idata);        
          cudaMemcpy(h_bools, d_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
          scan(n, h_indicies, h_bools);
          cudaMemcpy(d_indicies, h_indicies, n * sizeof(int), cudaMemcpyHostToDevice);
          kernScatter<<<blocksPerGrid, BLOCK_SIZE >>>(n, d_odata, d_idata, d_bools, d_indicies);

          timer().endGpuTimer();
          compactTest = false;

          cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
          int count = 0;
          for (int i = 0; i < n; i++)
            if (odata[i] != 0)
              count++;
          cudaFree(d_bools); cudaFree(d_idata); cudaFree(d_indicies); cudaFree(d_odata);
          delete[] h_indicies, h_bools;
          return count;
        }
    }
}
