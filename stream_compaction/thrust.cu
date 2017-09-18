#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

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
          int *dev_idata;
          int *dev_odata;

          cudaMalloc((void **)&dev_idata, n * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_idata!!!");

          cudaMalloc((void **)&dev_odata, n * sizeof(int));
          checkCUDAErrorWithLine("malloc dev_odata!!!");

          cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAErrorWithLine("memcpy dev_idata from host!!!");

          thrust::device_ptr<int> dev_thrust_idata(dev_idata);
          thrust::device_ptr<int> dev_thrust_odata(dev_odata);

          // pass in cpu pointers here
          thrust::device_vector<int> dev_vector_idata(idata, idata + n);
          thrust::device_vector<int> dev_vector_odata(odata, odata + n);

          timer().startGpuTimer();
          thrust::exclusive_scan(idata, idata + n, odata);//dev_vector_idata.begin(), dev_vector_idata.end(), dev_vector_odata.begin());
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
          timer().endGpuTimer();

          //cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
          //checkCUDAErrorWithLine("memcpy dev_odata to host!!!");

          cudaFree(dev_idata);
          checkCUDAErrorWithLine("free dev_idata!!!");

          cudaFree(dev_odata);
          checkCUDAErrorWithLine("free dev_odata!!!");
        }
    }
}
