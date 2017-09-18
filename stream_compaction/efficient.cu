#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernWorkEfficientScanUpSweep(int n, int d, int* odata, const int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int twoPowDPlusOne = (int)powf(2, d + 1);
            index *= twoPowDPlusOne;
            if (index >= n)
            {
                return;
            }

            odata[index + twoPowDPlusOne - 1] += idata[index + ((int)powf(2, d)) - 1];
        }

        __global__ void kernWorkEfficientScanDownSweep(int n, int d, int* odata, const int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int twoPowDPlusOne = (int)powf(2, d + 1);
            index *= twoPowDPlusOne;
            if (index >= n || n == 1)
            {
                return;
            }

            int twoPowD = (int)powf(2, d);
            int t = idata[index + twoPowD - 1];
            odata[index + twoPowD - 1] = idata[index + twoPowDPlusOne - 1];
            odata[index + twoPowDPlusOne - 1] += t;
        }

        __global__ void kernSetFinalValue(int n, int* odata, const int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }

            if (index == n - 1)
            {
                odata[n - 1] = 0;
                return;
            }

            odata[index] = idata[index];
        }

        __global__ void kernSetTempArray(int n, int* odata, const int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }

            odata[index] = (idata[index]) ? 1 : 0;
        }

        __global__ void kernScatter(int n, int* odata, const int* tempArray, const int* scannedTempArray, const int* idata)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }

            if (tempArray[index])
            {
                odata[scannedTempArray[index]] = idata[index];
            }
            else
            {
                return;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int nLog2 = ilog2ceil(n);

            // Pad the array with zeroes in the event that it isn't a power of two
            int newN = pow(2, nLog2);
            int* paddedArray = (int*) malloc(sizeof(int) * newN);

            for (int i = 0; i < newN; ++i)
            {
                if (i < newN - n)
                {
                    paddedArray[i] = 0;
                }
                else
                {
                    paddedArray[i] = idata[i - newN + n];
                }
            }

            int numArrayBytes = sizeof(int) * newN;

            // Global memory arrays
            int* dev_idata;
            int* dev_odata;

            // Kernel configuration
            float blockSize = 64.f;
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((((float) newN) + blockSize - 1.f) / blockSize);

            // CUDA Mallocs
            cudaMalloc((void**)&dev_odata, numArrayBytes);
            checkCUDAError("cudaMalloc dev_odata failed!", __LINE__);

            cudaMalloc((void**)&dev_idata, numArrayBytes);
            checkCUDAError("cudaMalloc dev_idata failed!", __LINE__);

            // Copy the input data array to the device
            cudaMemcpy(dev_idata, paddedArray, numArrayBytes, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy failed!", __LINE__);

            // Copy the input data array to the output array (needs the initial data to be correct)
            cudaMemcpy(dev_odata, dev_idata, numArrayBytes, cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy failed!", __LINE__);

            timer().startGpuTimer();

            // Perform the work efficient scan
            for (int d = 0; d <= nLog2 - 1; ++d)
            {
                kernWorkEfficientScanUpSweep << < fullBlocksPerGrid, threadsPerBlock >> > (newN, d, dev_odata, dev_idata);
                cudaMemcpy(dev_idata, dev_odata, numArrayBytes, cudaMemcpyDeviceToDevice);
                checkCUDAError("memcpy failed!1", __LINE__);
            }

            // Set the final value of the array to 0 as the first step of the down sweep
            kernSetFinalValue << < fullBlocksPerGrid, threadsPerBlock >> > (newN, dev_odata, dev_idata);
            cudaMemcpy(dev_idata, dev_odata, numArrayBytes, cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy failed!2", __LINE__);

            for (int d = nLog2 - 1; d >= 0; --d)
            {
                kernWorkEfficientScanDownSweep << < fullBlocksPerGrid, threadsPerBlock >> > (newN, d, dev_idata, dev_odata);
                cudaMemcpy(dev_odata, dev_idata, numArrayBytes, cudaMemcpyDeviceToDevice);
                checkCUDAError("memcpy failed!3", __LINE__);
            }

            timer().endGpuTimer();

            cudaMemcpy(paddedArray, dev_odata, numArrayBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy failed4!", __LINE__);

            for (int i = 0; i < n; ++i)
            {
                odata[i] = paddedArray[i + newN - n];
            }

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            int nLog2 = ilog2ceil(n);

            // Pad the array with zeroes in the event that it isn't a power of two
            int newN = pow(2, nLog2);
            int* paddedArray = (int*) malloc(sizeof(int) * newN);

            for (int i = 0; i < newN; ++i)
            {
                if (i < newN - n)
                {
                    paddedArray[i] = 0;
                }
                else
                {
                    paddedArray[i] = idata[i - newN + n];
                }
            }

            int numArrayBytes = sizeof(int) * newN;

            // Global memory arrays
            int* dev_idata;
            int* dev_odata;
            int* dev_tempArray;
            int* dev_sumTempArray;

            // Kernel configuration
            float blockSize = 64.f;
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((((float)newN) + blockSize - 1.f) / blockSize);

            // CUDA Mallocs
            cudaMalloc((void**)&dev_odata, numArrayBytes);
            checkCUDAError("cudaMalloc dev_odata failed!", __LINE__);

            cudaMalloc((void**)&dev_idata, numArrayBytes);
            checkCUDAError("cudaMalloc dev_idata failed!", __LINE__);

            cudaMalloc((void**)&dev_tempArray, numArrayBytes);
            checkCUDAError("cudaMalloc dev_tempArray failed!", __LINE__);

            cudaMalloc((void**)&dev_sumTempArray, numArrayBytes);
            checkCUDAError("cudaMalloc dev_sumTempArray failed!", __LINE__);

            // Copy the input data array to the device
            cudaMemcpy(dev_idata, paddedArray, numArrayBytes, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy failed!", __LINE__);

            kernSetTempArray << < fullBlocksPerGrid, threadsPerBlock >> > (newN, dev_tempArray, dev_idata);

            // Copy the input data array to the output array (needs the initial data to be correct)
            cudaMemcpy(dev_sumTempArray, dev_tempArray, numArrayBytes, cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy failed!", __LINE__);

            // Copy the input data array to the output array (needs the initial data to be correct)
            cudaMemcpy(dev_idata, dev_tempArray, numArrayBytes, cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy failed!", __LINE__);

            timer().startGpuTimer();

            // Perform the work efficient scan - up sweep
            for (int d = 0; d <= nLog2 - 1; ++d)
            {
                dim3 fullBlocksPerGrid_efficient((((float) newN) / pow(2, d + 1) + blockSize - 1.f) / blockSize);
                kernWorkEfficientScanUpSweep << < fullBlocksPerGrid, threadsPerBlock >> > (newN, d, dev_sumTempArray, dev_idata);
                cudaMemcpy(dev_idata, dev_sumTempArray, numArrayBytes, cudaMemcpyDeviceToDevice);
                checkCUDAError("memcpy failed!1", __LINE__);
            }

            // Set the final value of the array to 0 as the first step of the down sweep
            kernSetFinalValue << < fullBlocksPerGrid, threadsPerBlock >> > (newN, dev_sumTempArray, dev_idata);
            cudaMemcpy(dev_idata, dev_sumTempArray, numArrayBytes, cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy failed!2", __LINE__);

            // Perform the work efficient scan - up sweep
            for (int d = nLog2 - 1; d >= 0; --d)
            {
                dim3 fullBlocksPerGrid_efficient((((float) newN) / pow(2, d + 1) + blockSize - 1.f) / blockSize);
                kernWorkEfficientScanDownSweep << < fullBlocksPerGrid, threadsPerBlock >> > (newN, d, dev_sumTempArray, dev_idata);
                cudaMemcpy(dev_idata, dev_sumTempArray, numArrayBytes, cudaMemcpyDeviceToDevice);
                checkCUDAError("memcpy failed!3", __LINE__);
            }

            cudaMemcpy(dev_idata, paddedArray, numArrayBytes, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy failed!", __LINE__);

            // Scatter
            kernScatter << < fullBlocksPerGrid, threadsPerBlock >> > (newN, dev_odata, dev_tempArray, dev_sumTempArray, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(paddedArray, dev_sumTempArray, numArrayBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy failed4!", __LINE__);

            int compactSize = paddedArray[newN - 1];

            cudaMemcpy(paddedArray, dev_tempArray, numArrayBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy failed4!", __LINE__);

            if (paddedArray[newN - 1])
            {
                compactSize++;
            }

            cudaMemcpy(paddedArray, dev_odata, numArrayBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy failed4!", __LINE__);

            for (int i = 0; i < n; ++i)
            {
                odata[i] = paddedArray[i];
            }

            cudaFree(dev_idata);
            cudaFree(dev_tempArray);
            cudaFree(dev_sumTempArray);
            cudaFree(dev_odata);
            return compactSize;
        }
    }
}
