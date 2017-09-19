#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
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

    int *dev_idata;
    int *dev_odata;

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    void scan(int n, int *odata, const int *idata) {
      cudaMalloc((void**)&dev_idata, n * sizeof(int));
      cudaMalloc((void**)&dev_odata, n * sizeof(int));

      cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

      thrust::device_ptr<int> dev_thrust_idata(dev_idata);
      thrust::device_ptr<int> dev_thrust_odata(dev_odata);

      timer().startGpuTimer();
      thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);
      timer().endGpuTimer();

      cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(dev_idata);
      cudaFree(dev_odata);
    }
  }
}
