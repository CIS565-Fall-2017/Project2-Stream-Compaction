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

    __global__ void kernScan(int n, int *dev_input, int *dev_output1, int *dev_output2)
    {
      int index = (blockIdx.x * blockDim.x) + threadIdx.x;

      if (index >= n)
      {
        return;
      }

      dev_output1[index] = index > 0 ? dev_input[index - 1] : 0;
      __syncthreads();

      for (int d = 1; d <= ilog2ceil(n); d++)
      {

        int offset = pow(2, d - 1);
        if (index >= offset)
        {
          dev_output2[index] = dev_output1[index - offset] + dev_output1[index];
        }
        else
        {
          dev_output2[index] = dev_output1[index];
        }

        int *temp = dev_output1;
        dev_output1 = dev_output2;
        dev_output2 = temp;
        __syncthreads();
      }
    }

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int *odata, const int *idata) {
      int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      cudaMalloc((void**)dev_input, n * sizeof(int));
      cudaMalloc((void**)&dev_output1, n * sizeof(int));
      cudaMalloc((void**)&dev_output2, n * sizeof(int));

      cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

      timer().startGpuTimer();
      kernScan << <numBlocks, BLOCK_SIZE >> > (n, dev_input, dev_output1, dev_output2);
      timer().endGpuTimer();

      cudaMemcpy(odata, dev_output1, n * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(dev_input);
      cudaFree(dev_output1);
      cudaFree(dev_output2);
    }
  }
}
