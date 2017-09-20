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
            // Create host vector
			thrust::host_vector<int> hostVector(n);
			thrust::copy(idata, idata + n, hostVector.begin());

			// Create device vectors
			thrust::device_vector<int> inDeviceVector = hostVector;
			thrust::device_vector<int> outDeviceVector(n);

			// Scan
            timer().startGpuTimer();
			thrust::exclusive_scan(inDeviceVector.begin(), inDeviceVector.end(), outDeviceVector.begin());
			timer().endGpuTimer();

			// Copy data from device
			thrust::copy(outDeviceVector.begin(), outDeviceVector.end(), odata);
        }
    }
}
