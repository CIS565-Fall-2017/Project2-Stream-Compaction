#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		__global__ void EfficientScanUpSweep(int* idata, int n, int step1,int step2)
		{

			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if ((index >= n) || (index < 0))
			{
				return;
			}
			if ((index%step2 == 0) || (index == 0))
			{
				idata[index + step2 - 1] += idata[index + step1 - 1];
			}

		}

		__global__ void EfficientScanDownSweep(int* idata, int n, int step1, int step2)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if ((index >= n) || (index < 0))
			{
				return;
			}
			if ((index%step2 == 0) || (index == 0))
			{
				int t = idata[index + step1 - 1];
				idata[index + step1 - 1] = idata[index + step2 - 1];
				idata[index + step2 - 1] += t;
			}
		}

		__global__ void AssignInitial(int* idata, int n)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index == (n - 1))
			{
				idata[index] = 0;
			}
			return;
		}

		__global__ void AssignExtra(int size, int* extraArray)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if ((index >= size) || (index < 0))
			{
				return;
			}
			extraArray[index] = 0;
		}

		__global__ void EfficientScanDownSweepNonPower(int n, int* idata, int* extraArray, int step1, int step2)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if ((index >= n) || (index < 0))
			{
				return;
			}

			if ((index%step2 == 0) || (index == 0))
			{
				if (((index + step2 - 1) > n - 1) && ((index + step1 - 1) > n - 1)) 
				{
					int extraIndex2 = index + step2 - 1 - n;
					int extraIndex1 = index + step1 - 1 - n;
					int t = extraArray[extraIndex1];
					extraArray[extraIndex1] = extraArray[extraIndex2];
					extraArray[extraIndex2] += t;
				}
				else if((index + step2 - 1) > n - 1)
				{
					int extraIndex = index + step2 - 1 - n;
					int t = idata[index + step1 - 1];
					idata[index + step1 - 1] = extraArray[extraIndex];
					extraArray[extraIndex] += t;
				}
				else
				{
					int t = idata[index + step1 - 1];
					idata[index + step1 - 1] = idata[index + step2 - 1];
					idata[index + step2 - 1] += t;
				}
			}
		}

        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int blockSize = 256;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			const int memoryCopySize = n * sizeof(int);
			int step1,step2;

			//false means output is odata, true means output is idata
			//bool outAndInFlag = false;

			int* dev_idata;
			int* dev_odata;

			cudaMalloc((void**)&dev_idata, memoryCopySize);
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_idata, idata, memoryCopySize, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");

			cudaMalloc((void**)&dev_odata, memoryCopySize);
			checkCUDAError("cudaMalloc dev_odata failed!");

			for (int d = 0;d <= (ilog2ceil(n)-1);d++)
			{
				step1 = pow(2, d);
				step2 = pow(2, d + 1);
				EfficientScanUpSweep << <fullBlocksPerGrid, blockSize >> > (dev_idata, n, step1,step2);
				cudaThreadSynchronize();
			}

			//the ideal case of power-of-two
			if (pow(2, ilog2ceil(n)) == n)
			{
				AssignInitial << <fullBlocksPerGrid, blockSize >> > (dev_idata, n);
				cudaThreadSynchronize();

				for (int d = (ilog2ceil(n) - 1);d >= 0;d--)
				{
					step1 = pow(2, d);
					step2 = pow(2, d + 1);
					EfficientScanDownSweep << <fullBlocksPerGrid, blockSize >> >(dev_idata, n, step1, step2);
					cudaThreadSynchronize();
				}
			}
			//the non-power-of-two cases
			else
			{
				int numberToAdd = pow(2, ilog2ceil(n)) - n;
				int* extraArray;
				cudaMalloc((void**)&extraArray, numberToAdd * sizeof(int));

				AssignExtra << <fullBlocksPerGrid, blockSize >> >(numberToAdd, extraArray);
				cudaDeviceSynchronize();
				for (int d = (ilog2ceil(n) - 1);d >= 0;d--)
				{
					step1 = pow(2, d);
					step2 = pow(2, d + 1);
					EfficientScanDownSweepNonPower << <fullBlocksPerGrid, blockSize >> >(n, dev_idata, extraArray, step1, step2);
					cudaThreadSynchronize();
				}		

				cudaFree(extraArray);
			}		

			cudaMemcpy(odata, dev_idata, memoryCopySize, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			
			cudaFree(dev_idata);
			cudaFree(dev_odata);

            timer().endGpuTimer();
        }


		__global__ void EfficientMappingAlgorithm(int n, int* idata, int * mappedData)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if ((index >= n) || (index < 0))
			{
				return;
			}
			mappedData[index] = 0;

			if (idata[index] != 0)
			{
				mappedData[index] = 1;
			}
		}


		__global__ void EfficientCompactAlgorithm(int n, int* idata, int *odata,int* mappedData,int* scannedData)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if ((index >= n) || (index < 0))
			{
				return;
			}

			if (mappedData[index] != 0)
			{
				int idx = scannedData[index];
				odata[idx] = idata[index];
			}
		}

		__global__ void ODataInitialize(int n, int*odata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if ((index >= n) || (index < 0))
			{
				return;
			}

			odata[index] = -1;
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
            timer().startGpuTimer();
            // TODO
			int blockSize = 256;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			const int memoryCopySize = n * sizeof(int);

			int step1,step2,count=0;

			int* scannedDataIn;
			int* mappedData;
			int* dev_odata;
			int* dev_idata;

			cudaMalloc((void**)&scannedDataIn, memoryCopySize);
			checkCUDAError("cudaMalloc scannedDataIn failed!");

			cudaMalloc((void**)&mappedData, memoryCopySize);
			checkCUDAError("cudaMalloc mappedData failed!");
			
			cudaMalloc((void**)&dev_odata, memoryCopySize);
			checkCUDAError("cudaMalloc dev_odata failed!");
			
			cudaMalloc((void**)&dev_idata, memoryCopySize);
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMemcpy(dev_idata, idata, memoryCopySize, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");

			ODataInitialize << <fullBlocksPerGrid, blockSize >> >(n, dev_odata);
			cudaDeviceSynchronize();

			EfficientMappingAlgorithm << <fullBlocksPerGrid, blockSize >> > (n, dev_idata, mappedData);
			cudaDeviceSynchronize();

			cudaMemcpy(scannedDataIn, mappedData, memoryCopySize, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy scannedDataIn failed!");

			for (int d = 0;d <= (ilog2ceil(n) - 1);d++)
			{
				step1 = pow(2, d);
				step2 = pow(2, d + 1);
				EfficientScanUpSweep << <fullBlocksPerGrid, blockSize >> > (scannedDataIn, n, step1, step2);
				cudaThreadSynchronize();
			}

			if (pow(2, ilog2ceil(n)) == n)
			{
				AssignInitial << <fullBlocksPerGrid, blockSize >> > (scannedDataIn, n);
				cudaThreadSynchronize();

				for (int d = (ilog2ceil(n) - 1);d >= 0;d--)
				{
					step1 = pow(2, d);
					step2 = pow(2, d + 1);
					EfficientScanDownSweep << <fullBlocksPerGrid, blockSize >> >(scannedDataIn, n, step1, step2);
					cudaThreadSynchronize();
				}

				EfficientCompactAlgorithm << <fullBlocksPerGrid, blockSize >> > (n, dev_idata, dev_odata, mappedData, scannedDataIn);
				cudaThreadSynchronize();

				cudaMemcpy(odata, dev_odata, memoryCopySize, cudaMemcpyDeviceToHost);

				int value = odata[count];

				while (value != -1)
				{
					count++;
					value = odata[count];
				}


			}
			//the non-power-of-two cases
			else
			{
				int numberToAdd = pow(2, ilog2ceil(n)) - n;
				int* extraArray;
				cudaMalloc((void**)&extraArray, numberToAdd * sizeof(int));

				AssignExtra << <fullBlocksPerGrid, blockSize >> >(numberToAdd, extraArray);
				cudaDeviceSynchronize();
				for (int d = (ilog2ceil(n) - 1);d >= 0;d--)
				{
					step1 = pow(2, d);
					step2 = pow(2, d + 1);
					EfficientScanDownSweepNonPower << <fullBlocksPerGrid, blockSize >> >(n, scannedDataIn, extraArray, step1, step2);
					cudaThreadSynchronize();
				}

				cudaFree(extraArray);


				EfficientCompactAlgorithm << <fullBlocksPerGrid, blockSize >> > (n, dev_idata, dev_odata, mappedData, scannedDataIn);
				cudaThreadSynchronize();

				cudaMemcpy(odata, dev_odata, memoryCopySize, cudaMemcpyDeviceToHost);

				int value = odata[count];

				while (value != -1)
				{
					count++;
					value = odata[count];
				}

			}

			
			cudaFree(scannedDataIn);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(mappedData);

            timer().endGpuTimer();
			return count;
        }
    }
}
