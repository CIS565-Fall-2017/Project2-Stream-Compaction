#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void kernNaiveScan(int *g_odata, const int *g_idata, int n, int offset) {

          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) return;
          if (index >= offset)
            g_odata[index] = g_idata[index - offset] + g_idata[index];
          else
            g_odata[index] = g_idata[index];
        }

        __global__ void kernExclusive(int *g_odata, const int *g_idata, int n) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) return;
          g_odata[index] = index == 0 ? 0 : g_idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
          int *d_idata, *d_odata;
          cudaMalloc((void**)&d_idata, n * sizeof(int));
          cudaMalloc((void**)&d_odata, n * sizeof(int));
          cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          timer().startGpuTimer();
            
          for (int d = 1; d <= ilog2ceil(n); d++) {
            kernNaiveScan<<<blocksPerGrid, BLOCK_SIZE>>>(d_odata, d_idata, n, 1 << d - 1);
            std::swap(d_idata, d_odata);
          }
          kernExclusive<<<blocksPerGrid, BLOCK_SIZE>>>(d_odata, d_idata, n);

          timer().endGpuTimer();

          cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
          cudaFree(d_idata);
          cudaFree(d_odata);
        }
    }
}
