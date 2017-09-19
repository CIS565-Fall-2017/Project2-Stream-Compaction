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
//            timer().startGpuTimer();
			int *dev_data;
			cudaMalloc((void**)&dev_data, sizeof(int) * n);
			cudaMemcpy((void*)dev_data, (const void*)idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			thrust::device_ptr<int> dev_thrust_data(dev_data);
            timer().startGpuTimer();
			thrust::exclusive_scan(dev_thrust_data, dev_thrust_data + n, dev_thrust_data);
			timer().endGpuTimer();
			cudaMemcpy((void*)odata, (const void*)dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_data);
//           timer().endGpuTimer();
        }
    }
}
