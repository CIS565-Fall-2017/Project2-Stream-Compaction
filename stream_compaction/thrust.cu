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
        void scan(int n, int *odata, const int *idata) 
		{
			thrust::device_vector<int> in(idata, idata + 1);
			thrust::device_vector<int> out(odata, odata + 1);
			//timer().startGpuTimer();
			//thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			//timer().endGpuTimer();
			//Only time the actual exclusive scan and not the memory copies that go with it
			timer().startGpuTimer();
			thrust::exclusive_scan(in.begin(), in.end(), out.begin());
			timer().endGpuTimer();
			//Important Reference for creating thrust device vectors: 
			//https://stackoverflow.com/questions/9495599/thrust-how-to-create-device-vector-from-host-array
			
			//create device vectors for thrust using the CPU side arrays idata and odata
			thrust::device_vector<int> dv_in(idata, idata + n);
			thrust::device_vector<int> dv_out(odata, odata + n);
			//timer().startGpuTimer();
			//thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			//timer().endGpuTimer();
			//Only time the actual exclusive scan and not the memory copies that go with it
			timer().startGpuTimer();
				thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			timer().endGpuTimer();

			//copy thrust out vector back into odata on cpu side
			thrust::copy(dv_out.begin(), dv_out.end(), odata);
        }
    }
}
