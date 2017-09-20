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

			int* a;
			cudaMalloc((void**)&a, n * sizeof(int));
			cudaMemcpy(a, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from CPU to GPU
			
			thrust::device_vector<int> in(n+1);
			thrust::device_vector<int> out(n+1);

			thrust::device_ptr<int> here(a);
			thrust::copy(here, here + n, in.begin()); //CPOY from GPU to Thrust

			//Start Timing
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

			thrust::exclusive_scan(in.begin(), in.end(), out.begin());
			timer().endGpuTimer();
			//End Timing

			thrust::copy(out.begin()+1, out.end(), odata);	//COPY from Thrust to CPU

			cudaFree(a);
			
			
        }
    }
}
