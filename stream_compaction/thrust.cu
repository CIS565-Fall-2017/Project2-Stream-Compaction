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
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:

			int *dev_idata, *dev_odata;
			dev_idata = nullptr;
			dev_odata = nullptr;

			cudaMalloc(&dev_idata, n * sizeof(int));
			cudaMalloc(&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			thrust::device_ptr<int> thrust_idata(dev_idata);
			thrust::device_ptr<int> thrust_odata(dev_odata);

			timer().startGpuTimer();
			thrust::exclusive_scan(thrust_idata, thrust_idata + n, thrust_odata);
            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
