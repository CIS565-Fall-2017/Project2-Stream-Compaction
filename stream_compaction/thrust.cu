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
			thrust::device_vector<int> thrust_dev_odata = thrust::device_vector<int>(odata, odata + n);
			thrust::device_vector<int> thrust_dev_idata = thrust::device_vector<int>(idata, idata + n);

            timer().startGpuTimer();
			thrust::exclusive_scan(thrust_dev_idata.begin(),thrust_dev_idata.end(), thrust_dev_odata.begin());
			timer().endGpuTimer();

			thrust::copy(thrust_dev_odata.begin(), thrust_dev_odata.end(), odata);
        }
    }
}
