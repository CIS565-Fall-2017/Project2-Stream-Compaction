#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int powTwo, int *odata, const int* idata)
        {
          int index = threadIdx.x + blockIdx.x * blockDim.x;
          if (index >= n)
          {
              return;
          }

          if (index >= powTwo)
          {
              odata[index] = idata[index - powTwo] + idata[index];
          }
          else
          {
              odata[index] = idata[index];
          }
        }

        __global__ void kernExclToInclScan(int n, int* odata, const int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
              return;
            }

            if (index == 0)
            {
                odata[index] == 0;
                return;
            }

            odata[index] = idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            // Global memory arrays
            int* dev_parallelArrayA;
            int* dev_parallelArrayB;
            int* dev_odata;

            int numArrayBytes = sizeof(int) * n;

            // CUDA Mallocs
            cudaMalloc((void**)&dev_odata, numArrayBytes);
            checkCUDAError("cudaMalloc dev_odata failed!", __LINE__);

            cudaMalloc((void**)&dev_parallelArrayA, numArrayBytes);
            checkCUDAError("cudaMalloc dev_parallelArrayA failed!", __LINE__);

            cudaMalloc((void**)&dev_parallelArrayB, numArrayBytes);
            checkCUDAError("cudaMalloc dev_parallelArrayB failed!", __LINE__);

            // Copy the input data array to the device
            cudaMemcpy(dev_parallelArrayA, idata, numArrayBytes, cudaMemcpyHostToDevice);

            // Kernel configuration
            float blockSize = 64.f;
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((((float) n) + blockSize - 1.f) / blockSize);
            int nLog2 = ilog2ceil(n);
            bool isReadingFromA = true;

            timer().startGpuTimer();
            
            for (int d = 1; d <= nLog2; ++d)
            {
              int powTwo = pow(2, d - 1);
              if (isReadingFromA)
              {
                  // Perform the *INCLUSIVE* scan
                  kernNaiveScan << < fullBlocksPerGrid, threadsPerBlock >> > (n, powTwo, dev_parallelArrayB, dev_parallelArrayA);
                  cudaMemcpy(dev_parallelArrayA, dev_parallelArrayB, numArrayBytes, cudaMemcpyDeviceToDevice);
              }
              else
              {
                  // Perform the scan
                  kernNaiveScan << < fullBlocksPerGrid, threadsPerBlock >> > (n, powTwo, dev_parallelArrayA, dev_parallelArrayB);
                  cudaMemcpy(dev_parallelArrayB, dev_parallelArrayA, numArrayBytes, cudaMemcpyDeviceToDevice);
              }
              isReadingFromA = !isReadingFromA;
            }

            timer().endGpuTimer();

            // Copy the output data back into the host array
            if (isReadingFromA)
            {
                kernExclToInclScan << < fullBlocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_parallelArrayA);
                cudaMemcpy(odata, dev_odata, numArrayBytes, cudaMemcpyDeviceToHost);
            }
            else
            {
                kernExclToInclScan << < fullBlocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_parallelArrayB);
                cudaMemcpy(odata, dev_odata, numArrayBytes, cudaMemcpyDeviceToHost);
            }

            // CUDA Frees
            cudaFree(dev_parallelArrayA);
            cudaFree(dev_parallelArrayB);
            cudaFree(dev_odata);
        }
    }
}
