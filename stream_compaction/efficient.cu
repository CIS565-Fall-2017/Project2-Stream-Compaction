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

		__global__ void kernScanUp(int n, int i, int* dev_idata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
			{
				return;
			}
			int p = 1 << (i);
			int p2 = 1 << (i+1);
			if (index % p2 == 0)
			{
				dev_idata[index + p2 - 1] += dev_idata[index + p - 1];
			}
		}

		__global__ void kernScanDown(int n, int i, int* dev_idata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
			{
				return;
			}
			int p = 1 << (i);
			int p2 = 1 << (i + 1);
			//index = index - 1;
			if (index % p2 == 0)
			{
				int t = dev_idata[index + p -1];
				dev_idata[index + p - 1] = dev_idata[index + p2 - 1];
				dev_idata[index + p2 - 1] += t;
			}
		}

		void kernScan(int m, int o, int* dev_idata)
		{
			dim3 fullBlocksPerGrid((o + blockSize - 1) / blockSize);
			for (int i = 0; i < m; i++)
			{
				kernScanUp << <fullBlocksPerGrid, blockSize >> >(o, i, dev_idata);
			}
			cudaMemset(dev_idata+o - 1, 0, sizeof(int));
			for (int i = m - 1; i > -1; i--)
			{
				kernScanDown << <fullBlocksPerGrid, blockSize >> >(o, i, dev_idata);
			}
		}

		__global__ void newkernScanUp(int n, int i, int* dev_idata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
			{
				return;
			}
			int p = 1 << (i);
			int p2 = 1 << (i + 1);
			dev_idata[index*p2 + p2 - 1] += dev_idata[index*p2 + p - 1];
		}

		__global__ void newkernScanDown(int n, int i, int* dev_idata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
			{
				return;
			}
			int p = 1 << (i);
			int p2 = 1 << (i + 1);
			int t = dev_idata[index*p2 + p - 1];
			dev_idata[index*p2 + p - 1] = dev_idata[index*p2 + p2 - 1];
			dev_idata[index*p2 + p2 - 1] += t;
		}

		void newkernScan(int m, int o, int* dev_idata)
		{
			dim3 fullBlocksPerGrid;//((o + blockSize - 1) / blockSize);
			int inter;
			for (int i = 0; i < m; i++)
			{
				inter = 1 << (i + 1);
				fullBlocksPerGrid = ((o / inter + blockSize - 1) / blockSize);
				newkernScanUp << <fullBlocksPerGrid, blockSize >> >(o / inter, i, dev_idata);
			}
			cudaMemset(dev_idata + o - 1, 0, sizeof(int));
			for (int i = m - 1; i > -1; i--)
			{
				inter = 1 << (i + 1);
				fullBlocksPerGrid = ((o / inter + blockSize - 1) / blockSize);
				newkernScanDown << <fullBlocksPerGrid, blockSize >> >(o / inter, i, dev_idata);
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
			int *dev_idata, *idataall;
			int m = ilog2ceil(n);
			int o = 1 << (m);
			idataall = (int*)malloc(o * sizeof(int));
			for (int i = 0; i < n; i++)
			{
				idataall[i] = idata[i];
			}
			for (int i = n; i < o; i++)
			{
				idataall[i] = 0;
			}
			cudaMalloc((void**)&dev_idata, o * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMemcpy(dev_idata, idataall, o * sizeof(int), cudaMemcpyHostToDevice);
			
			timer().startGpuTimer();

			newkernScan(m, o, dev_idata);

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_idata, o * sizeof(int), cudaMemcpyDeviceToHost);
			//exclusive->inclusive
			/*for (int i = 1; i < n; i++) {
				odata[i-1] = idataall[i];
			}
			if (n != o)
			{
				odata[n - 1] = idataall[n];
			}
			else
			{
				odata[n - 1] = idataall[n - 1] + idata[n - 1];
			}*/
			free(idataall);
			cudaFree(dev_idata);
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
			int *dev_idata, *dev_bools, *dev_odata, *dev_indices, *idataall;
			int m = ilog2ceil(n);
			int o = 1 << (m);
			idataall = (int*)malloc(o * sizeof(int));
			for (int i = 0; i < n; i++)
			{
				idataall[i] = idata[i];
			}
			for (int i = n; i < o; i++)
			{
				idataall[i] = 0;
			}
			cudaMalloc((void**)&dev_idata, o * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_bools, o * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_odata, o * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_indices, o * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");

			cudaMemcpy(dev_idata, idataall, o * sizeof(int), cudaMemcpyHostToDevice);
			timer().startGpuTimer();
			dim3 fullBlocksPerGrid((o + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(o, dev_bools, dev_idata);
			cudaMemcpy(dev_indices, dev_bools, o * sizeof(int), cudaMemcpyDeviceToDevice);
			newkernScan(m, o, dev_indices);
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(o, dev_odata, dev_idata, dev_bools, dev_indices);
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, o * sizeof(int), cudaMemcpyDeviceToHost);

			int num = 0;
			for (int i = 0; i < n; i++)
			{
				if (odata[i])
				{
					num++;
				}
				else
				{
					break;
				}
			}

			free(idataall);
			cudaFree(dev_idata);
			cudaFree(dev_bools);
			cudaFree(dev_odata);
			cudaFree(dev_indices);
			return num;
			//return -1;
        }
    }
}
