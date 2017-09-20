#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

#define BLOCK_SIZE 128

namespace StreamCompaction {
namespace Naive {

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
  static PerformanceTimer timer;
  return timer;
}
// TODO: __global__
__global__ void naive_scan_impl(int n, int offset, int *odata, const int *idata) {
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int out_index = index - offset;
  if (out_index < 0) {
    odata[index] = idata[index];
  } else if (index < n){
    odata[index] = idata[index] + idata[out_index];
  }
}

__global__ void shift_impl(int n, int *odata, const int *idata) {
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index == 0) {
    odata[0] = 0;
  }
  if (index < n - 1) {
    odata[index + 1] = idata[index];
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int *dev_array_A;
  int *dev_array_B;
  cudaMalloc((void**)&dev_array_A, n * sizeof(int));
  cudaMalloc((void**)&dev_array_B, n * sizeof(int));
  cudaMemcpy(dev_array_B, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
  bool direction = false;

  timer().startGpuTimer();
  shift_impl<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, dev_array_A, dev_array_B);
  cudaMemcpy(dev_array_B, dev_array_A, sizeof(int), cudaMemcpyDeviceToDevice);

  for (int offset = 1; offset < n; offset *= 2) {
    if (direction) {
      naive_scan_impl<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, offset, dev_array_A, dev_array_B);
    } else {
      naive_scan_impl<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, offset, dev_array_B, dev_array_A);
    }
    direction = !direction;
  }
  timer().endGpuTimer();
  cudaMemcpy(odata, (!direction ? dev_array_A : dev_array_B), sizeof(int) * n, cudaMemcpyDeviceToHost);
  cudaFree(dev_array_A);
  cudaFree(dev_array_B);
}

} // namespace Naive
} // namespace StreamCompaction
