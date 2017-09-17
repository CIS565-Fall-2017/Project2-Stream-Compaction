#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "iostream"

#define BLOCK_SIZE 128

namespace StreamCompaction {
namespace Efficient {

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
  static PerformanceTimer timer;
  return timer;
}

__global__ void UpSweep(int n, int offset, int *data) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int first_index = offset*(index)*2 + offset - 1;
  int end_index = first_index + offset;
  if (end_index < n) {
    data[end_index] += data[first_index];
  }
}

__global__ void set_zero(int n, int *data) {
  data[n - 1] = 0;
}

__global__ void clear_array(int n, int*data) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    data[index] = 0;
  }
}

__global__ void DownSweep(int n, int offset, int *data) {
   int index = threadIdx.x + blockIdx.x * blockDim.x;
   if (index < n) {
     //int left_index = index + (offset >> 1) - 1;
     //int right_index = index + offset - 1;
     int left_index = offset*index*2 + offset - 1;
     int right_index = left_index + offset;
     if (left_index < n && right_index < n) {
       int temp = data[left_index];
       data[left_index] = data[right_index];
       data[right_index] += temp;
     }
   }
}

void scan_impl(int arr_length, int *dev_array) {
  dim3 fullBlocksPerGrid((arr_length + BLOCK_SIZE - 1) / BLOCK_SIZE);
  for (int offset = 1; offset < arr_length; offset *= 2) {
    UpSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(arr_length, offset, dev_array);
  }
  set_zero<<<dim3(1), dim3(1)>>>(arr_length, dev_array);
  for(int offset = arr_length/2; offset >= 1; offset = offset >> 1) {
    DownSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(arr_length, offset, dev_array);
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int arr_length = (1 << ilog2ceil(n));
  int *dev_array;
  cudaMalloc((void**)&dev_array, arr_length * sizeof(int));
  clear_array<<<fullBlocksPerGrid, BLOCK_SIZE>>>(arr_length, dev_array);
  cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

  timer().startGpuTimer();
  scan_impl(arr_length, dev_array);
  timer().endGpuTimer();
  cudaMemcpy(odata, dev_array, sizeof(int) * n, cudaMemcpyDeviceToHost);
  cudaFree(dev_array);
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
  dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int arr_length = (1 << ilog2ceil(n));
  int *dev_array_data;
  int *dev_array_bool;
  int *dev_array_indices;
  int *dev_array_out;

  cudaMalloc((void**)&dev_array_data, arr_length * sizeof(int));
  cudaMalloc((void**)&dev_array_bool, arr_length * sizeof(int));
  cudaMalloc((void**)&dev_array_indices, arr_length * sizeof(int));
  cudaMalloc((void**)&dev_array_out, arr_length * sizeof(int));

  clear_array<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, dev_array_data);
  cudaMemcpy(dev_array_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  timer().startGpuTimer();
  Common::kernMapToBoolean<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, dev_array_bool, dev_array_data);
  cudaMemcpy(dev_array_indices, dev_array_bool, arr_length * sizeof(int), cudaMemcpyDeviceToDevice);
  scan_impl(arr_length, dev_array_indices);
  Common::kernScatter<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, dev_array_out, dev_array_data, dev_array_bool, dev_array_indices);
  timer().endGpuTimer();
  cudaMemcpy(odata, dev_array_out, n*sizeof(int), cudaMemcpyDeviceToHost);
  int ret;
  int ret2;
  cudaMemcpy(&ret, dev_array_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ret2, dev_array_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
  return ret + ret2;
}

}
}
