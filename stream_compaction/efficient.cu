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


		__global__ void upSweep(int n, int d, int *idata) 
		{
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) 
			{
				return;
			}

			//based on slides
			int delta = 1 << d;
			int doubleDelta = 1 << (d + 1);
			
			if (index % doubleDelta == 0) 
			{
				idata[index + doubleDelta - 1] += idata[index + delta - 1];
			}
		}


		__global__ void downSweep(int n, int d, int *idata) 
		{
			//based on slides
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) 
			{
				return;
			}
			int delta = 1 << d;
			int doubleDelta = 1 << (d + 1);

			if (index % doubleDelta == 0) 
			{
				int t = idata[index + delta - 1];
				idata[index + delta - 1] = idata[index + doubleDelta - 1];
				idata[index + doubleDelta - 1] += t;
			}
		}

		//helper function for scan
		void helpscan(int n, int *devData) 
		{
			int blockNum = (n + blockSize - 1) / blockSize;

			for (int d = 0; d < ilog2ceil(n) - 1; d++) 
			{
				upSweep << <blockNum, blockSize >> >(n, d, devData);
			}

			int counter = 0;
			cudaMemcpy(&devData[n - 1], &counter, sizeof(int), cudaMemcpyHostToDevice);

			for (int d = ilog2ceil(n) - 1; d >= 0; d--) 
			{
				downSweep << <blockNum, blockSize >> >(n, d, devData);
			}
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// TODO
			//handle both conditions of PO2 and NPO2
			int num;
			int *t;
			int depth = ilog2ceil(n);

			if (n & (n - 1) != 0) 
			{
				num = 1 << depth;
				t = (int*)malloc(num * sizeof(int));
				memcpy(t, idata, num * sizeof(int));


				for (int j = n; j < num; j++) 
				{
					t[j] = 0;
				}

			}
			else 
			{
				num = n;
				t = (int*)malloc(num * sizeof(int));
				memcpy(t, idata, num * sizeof(int));
			}

			int size = num * sizeof(int);
			int *devi;

			cudaMalloc((void**)&devi, size);
			cudaMemcpy(devi, t, size, cudaMemcpyHostToDevice);

			timer().startGpuTimer();
			helpscan(num, devi);
			timer().endGpuTimer();

			cudaMemcpy(odata, devi, size, cudaMemcpyDeviceToHost);
			cudaFree(devi);
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
			// TODO
			//handle both conditions of PO2 and NPO2
			int num;
			int *t;
			int depth = ilog2ceil(n);

			if (n & (n - 1) != 0)
			{
				num = 1 << depth;
				t = (int*)malloc(num * sizeof(int));
				memcpy(t, idata, num * sizeof(int));


				for (int j = n; j < num; j++)
				{
					t[j] = 0;
				}

			}
			else
			{
				num = n;
				t = (int*)malloc(num * sizeof(int));
				memcpy(t, idata, num * sizeof(int));
			}

			int asize = num * sizeof(int);
			int blockNum = (num + blockSize - 1) / blockSize;
			int *devi;
			int *devo;
			int *devm;

			cudaMalloc((void**)&devi, asize);
			checkCUDAError("cudaMalloc  failed");
			cudaMalloc((void**)&devo, asize);
			checkCUDAError("cudaMalloc  failed");
			cudaMalloc((void**)&devm, asize);
			checkCUDAError("cudaMalloc  failed");

			timer().startGpuTimer();

			cudaMemcpy(devi, t, asize, cudaMemcpyHostToDevice);
			StreamCompaction::Common::kernMapToBoolean << <blockNum, blockSize >> >(num, devm, devi);

			int end;
			cudaMemcpy(&end, devm + num - 1, sizeof(int), cudaMemcpyDeviceToHost);

			helpscan(num, devm);

			int size;
			cudaMemcpy(&size, devm + num - 1, sizeof(int), cudaMemcpyDeviceToHost);

			StreamCompaction::Common::kernScatter << <blockNum, blockSize >> >(num, devo, devi, devm, devm);

			timer().endGpuTimer();

			cudaMemcpy(odata, devo, asize, cudaMemcpyDeviceToHost);


			if (end == 1) 
			{
				size++;
			}

			cudaFree(devi);
			cudaFree(devo);
			cudaFree(devm);

			return size;
        }
    }
}
