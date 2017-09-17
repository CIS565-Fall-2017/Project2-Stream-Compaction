#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
  static PerformanceTimer timer;
  return timer;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  thrust::host_vector<int> hv_in(idata, idata + n);
  thrust::device_vector<int> dv_in = hv_in;
  timer().startGpuTimer();
  thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_in.begin());
  timer().endGpuTimer();
  thrust::copy(dv_in.begin(), dv_in.end(), odata);
}

} // namespace Thrust
} // namespace StreamCompaction
