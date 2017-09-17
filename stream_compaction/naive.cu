#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
  namespace Naive {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }

    int *dev_input;
    int *dev_output1;
    int *dev_output2;

    const int BLOCK_SIZE = 128;

    __global__ void kernScan(int n, int offset, int *dev_input, int *dev_output1, int *dev_output2)
    {
      int index = (blockIdx.x * blockDim.x) + threadIdx.x;

      if (index >= n)
      {
        return;
      }

      if (index >= offset)
      {
        dev_output2[index] = dev_output1[index - offset] + dev_output1[index];
      }
      else
      {
        dev_output2[index] = dev_output1[index];
      }
    }

    __global__ void kernShift(int n, int *dev_input, int *dev_output)
    {
      int index = (blockIdx.x * blockDim.x) + threadIdx.x;

      if (index >= n)
      {
        return;
      }

      dev_output[index] = index > 0 ? dev_input[index - 1] : 0;
    }

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int *odata, const int *idata) {
      int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      int *temp;
      int d, offset;

      cudaMalloc((void**)&dev_input, n * sizeof(int));
      cudaMalloc((void**)&dev_output1, n * sizeof(int));
      cudaMalloc((void**)&dev_output2, n * sizeof(int));

      cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

      timer().startGpuTimer();

      kernShift << <numBlocks, BLOCK_SIZE >> > (n, dev_input, dev_output1);

      cudaMemcpy(odata, dev_output1, n * sizeof(int), cudaMemcpyDeviceToHost);

      for (d = 1; d <= ilog2ceil(n); d++)
      {
        offset = pow(2, d - 1);

        kernScan << <numBlocks, BLOCK_SIZE >> > (n, offset, dev_input, dev_output1, dev_output2);

        temp = dev_output1;
        dev_output1 = dev_output2;
        dev_output2 = temp;
      }

      timer().endGpuTimer();

      cudaMemcpy(odata, dev_output1, n * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(dev_input);
      cudaFree(dev_output1);
      cudaFree(dev_output2);
    }
  }
}