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

    int *dev_data;
    int *dev_odata;
    int *dev_bools;
    int *dev_indices;
    int *dev_idata;

    const int BLOCK_SIZE = 64;

    __global__ void kernUpSweep(int n, int width, int *data)
    {
      int index = (blockIdx.x * blockDim.x) + threadIdx.x;

      if (index >= n)
      {
        return;
      }

      data[(index + 1) * width - 1] += data[index * width - 1 + (width / 2)];
    }

    __global__ void kernDownSweep(int n, int width, int *data)
    {
      int index = (blockIdx.x * blockDim.x) + threadIdx.x;

      if (index >= n)
      {
        return;
      }

      int halfIndex = index * width - 1 + (width / 2);
      int fullIndex = (index + 1) * width - 1;
      int oldHalfIndexValue = data[halfIndex];

      data[halfIndex] = data[fullIndex];
      data[fullIndex] += oldHalfIndexValue;
    }

    __global__ void kernSetValueToZero(int i, int *data)
    {
      data[i] = 0;
    }

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int *odata, const int *idata) {
      int d, numThreads, numBlocks;

      int width = 1;
      int nPowerOfTwo = pow(2, ilog2ceil(n));
      int numIterations = ilog2(nPowerOfTwo) - 1;

      cudaMalloc((void**)&dev_data, nPowerOfTwo * sizeof(int));

      cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

      timer().startGpuTimer();

      for (d = 0; d <= numIterations; d++)
      {
        width *= 2;
        numThreads = nPowerOfTwo / width;
        numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernUpSweep << <numBlocks, BLOCK_SIZE >> > (numThreads, width, dev_data);
      }

      kernSetValueToZero << <1, 1 >> > (nPowerOfTwo - 1, dev_data);
      width = pow(2, numIterations + 2);

      for (d = numIterations; d >= 0; d--)
      {
        width /= 2;
        numThreads = nPowerOfTwo / width;
        numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kernDownSweep << <numBlocks, BLOCK_SIZE >> > (numThreads, width, dev_data);
      }

      timer().endGpuTimer();

      cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(dev_data);
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
      int size;
      int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

      cudaMalloc((void**)&dev_idata, n * sizeof(int));
      cudaMalloc((void**)&dev_bools, n * sizeof(int));
      cudaMalloc((void**)&dev_indices, n * sizeof(int));
      cudaMalloc((void**)&dev_odata, n * sizeof(int));

      cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

      //timer().startGpuTimer();

      Common::kernMapToBoolean << <numBlocks, BLOCK_SIZE>> > (n, dev_bools, dev_idata);
      cudaMemcpy(odata, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

      size = odata[n - 1];

      scan(n, odata, odata);

      size += odata[n - 1];

      cudaMemcpy(dev_indices, odata, n * sizeof(int), cudaMemcpyHostToDevice);
      Common::kernScatter << <numBlocks, BLOCK_SIZE >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

      //timer().endGpuTimer();

      cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(dev_idata);
      cudaFree(dev_bools);
      cudaFree(dev_indices);
      cudaFree(dev_odata);

      return size;
    }
  }
}
