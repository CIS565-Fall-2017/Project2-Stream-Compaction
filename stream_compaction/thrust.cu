#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"
#include "thrust/remove.h"
#include "thrust/execution_policy.h"
#include "thrust/copy.h"

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
		struct is_zero
		{
			__host__ __device__
				bool operator()(const int &x)
			{
				return (x == 0);
			}
		};
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

			thrust::host_vector<int> ho_in(n); 
			for (int i = 0;i < n;i++)
			{
				ho_in[i] = idata[i];
			}

			thrust::device_vector<int> dv_in = ho_in;
			thrust::device_vector<int> dv_out(n);
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			thrust::host_vector<int> ho_out = dv_out;

			for (int j = 0;j < n;j++)
			{
				odata[j] = ho_out[j];
			}
			
            timer().endGpuTimer();

			//remove_if

			//odata = thrust::remove_if(thrust::host, odata, odata + n, is_zero());

        }
    }
}
