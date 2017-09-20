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

        __global__ void kernWorkEfficientScanUpSweep(int n, int d, int* data)
        {
            /*int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int index = blockId * blockDim.x + threadIdx.x;*/
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int twoPowD = 1 << (d);
            int twoPowDPlusOne = 1 << (d + 1);
            
            if (index >= (n / twoPowDPlusOne))
            {
                return;
            }

            index = (index + 1) * twoPowDPlusOne - 1;
            int add = data[index - twoPowD];
            data[index] += add;
        }

        __global__ void kernWorkEfficientScanDownSweep(int n, int d, int* data)
        {
            /*int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int index = blockId * blockDim.x + threadIdx.x;*/
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int twoPowD = 1 << (d);
            int twoPowDPlusOne = 1 << (d + 1);
            
            if (index >= (n / twoPowDPlusOne))
            {
                return;
            }

            index = (index + 1) * twoPowDPlusOne - 1;
            
            int t = data[index - twoPowD];
            data[index - twoPowD] = data[index];
            data[index] += t;
        }

        __global__ void kernSetFinalValue(int n, int* data)
        {
            /*int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int index = blockId * blockDim.x + threadIdx.x;*/
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index == n - 1)
            {
                data[n - 1] = 0;
                return;
            }
        }

        __global__ void kernSetTempArray(int n, int* odata, const int* idata)
        {
            /*int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int index = blockId * blockDim.x + threadIdx.x;*/
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            odata[index] = (idata[index]) ? 1 : 0;
        }

        __global__ void kernScatter(int n, int* odata, const int* tempArray, const int* scannedTempArray, const int* idata)
        {
            /*int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int index = blockId * blockDim.x + threadIdx.x;*/
            int index = blockIdx.x * blockDim.x + threadIdx.x;
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
            int* dev_data;

            // Kernel configuration
            float blockSize = 512.f;
            dim3 threadsPerBlock(blockSize);
            float numBlocks = (((float)newN) + blockSize - 1.f) / blockSize;
            dim3 fullBlocksPerGrid((unsigned int) numBlocks);

            cudaMalloc((void**)&dev_data, numArrayBytes);
            checkCUDAError("cudaMalloc dev_idata failed!");

            // Copy the input data array to the device
            cudaMemcpy(dev_data, paddedArray, numArrayBytes, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy failed!");

            timer().startGpuTimer();

            // Perform the work efficient scan
            for (int d = 0; d <= nLog2 - 1; ++d)
            {
                if (n == 1) break;
                float numBlocksDynamic = (((float) newN) / pow(2, d + 1) + blockSize - 1.f) / blockSize;
                dim3 fullBlocksPerGrid_efficient((unsigned int) numBlocksDynamic);
                kernWorkEfficientScanUpSweep << < fullBlocksPerGrid, threadsPerBlock >> > (newN, d, dev_data);
            }

            // Set the final value of the array to 0 as the first step of the down sweep
            // Replace this is a device to host memcpy - ask Josh
            kernSetFinalValue << < fullBlocksPerGrid, threadsPerBlock >> > (newN, dev_data);

            for (int d = nLog2 - 1; d >= 0; --d)
            {
                if (n == 1) break;
                float numBlocksDynamic = (((float) newN) / pow(2, d + 1) + blockSize - 1.f) / blockSize;
                dim3 fullBlocksPerGrid_efficient((unsigned int) numBlocksDynamic);
                kernWorkEfficientScanDownSweep << < fullBlocksPerGrid, threadsPerBlock >> > (newN, d, dev_data);
            }

            timer().endGpuTimer();

            cudaMemcpy(paddedArray, dev_data, numArrayBytes, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy failed4!");

            // De-pad the zeroes from the array
            for (int i = 0; i < n; ++i)
            {
                odata[i] = paddedArray[i + newN - n];
            }

            cudaFree(dev_data);
            delete(paddedArray);
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

            timer().startGpuTimer();

            // Perform the work efficient scan - up sweep
            for (int d = 0; d <= nLog2 - 1; ++d)
            {
                dim3 fullBlocksPerGrid_efficient((((float) newN) / pow(2, d + 1) + blockSize - 1.f) / blockSize);
                kernWorkEfficientScanUpSweep << < fullBlocksPerGrid, threadsPerBlock >> > (newN, d, dev_sumTempArray);
            }

            // Set the final value of the array to 0 as the first step of the down sweep
            kernSetFinalValue << < fullBlocksPerGrid, threadsPerBlock >> > (newN, dev_sumTempArray);

            // Perform the work efficient scan - up sweep
            for (int d = nLog2 - 1; d >= 0; --d)
            {
                dim3 fullBlocksPerGrid_efficient((((float) newN) / pow(2, d + 1) + blockSize - 1.f) / blockSize);
                kernWorkEfficientScanDownSweep << < fullBlocksPerGrid, threadsPerBlock >> > (newN, d, dev_sumTempArray);
            }

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
