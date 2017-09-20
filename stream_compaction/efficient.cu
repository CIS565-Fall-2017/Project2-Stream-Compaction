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

		__global__ void upsweep(int n, int k, int* dev)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			if ((index % (2 * k) == 0) && (index + (2 * k) <= n))
				dev[index + (2 * k) - 1] += dev[index + k - 1];
		}

		__global__ void downsweep(int n, int k, int* dev)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			if ((index % (2 * k) == 0) && (index + (2 * k) <= n))
			{
				int tmp = dev[index + k - 1];
				dev[index + k - 1] = dev[index + (2 * k) - 1];
				dev[index + (2 * k) - 1] += tmp;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
       void scan(int n, int *odata, const int *idata) {

			int* dev;
			int potn = 1 << ilog2ceil(n);

			cudaMalloc((void**)&dev, potn * sizeof(int));
			checkCUDAError("Malloc for input device failed\n");

			cudaMemset(dev, 0, potn * sizeof(n));

			cudaMemcpy(dev, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy for device failed\n");

			dim3 fullBlocksPerGrid((potn + blockSize - 1) / blockSize);

			//timer().startGpuTimer();

			for (int k = 1; k < potn; k*=2)
			{
				upsweep <<< fullBlocksPerGrid, blockSize >>> (potn, k, dev);
			}

			cudaMemset(dev + potn - 1, 0, sizeof(int));

			for (int k = potn/2; k>0; k/=2)
			{
				downsweep <<< fullBlocksPerGrid, blockSize >>> (potn, k, dev);
			}

			//timer().endGpuTimer();

			cudaMemcpy(odata, dev, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy for output data failed\n");

			cudaFree(dev);
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
			int* idev;
			int* odev;
			cudaMalloc((void**)&idev, n * sizeof(int));
			checkCUDAError("cudaMalloc idata failed!");

			cudaMalloc((void**)&odev, n * sizeof(*odev));
			checkCUDAError("cudaMalloc odev failed!");

			cudaMemcpy(idev, idata, n * sizeof(*idata), cudaMemcpyHostToDevice);
			
			int potn = 1 << ilog2ceil(n);
			int* boolarr; 

			cudaMalloc((void**)&boolarr, potn * sizeof(int));
			checkCUDAError("cudaMalloc bool failed!");

			cudaMemset(boolarr, 0, potn * sizeof(int));

			int* indices;
			cudaMalloc((void**)&indices, potn * sizeof(int));
			checkCUDAError("cudaMalloc bool failed!");

			cudaMemcpy(indices, boolarr, n * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy from to dev_bools to dev_indices failed!");

			dim3 fullBlocksPerGrid((potn + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
			StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >>>(n, boolarr, idev);
			scan(n, indices, boolarr);
			StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize >>>(n, odev, idev, boolarr, indices);

            timer().endGpuTimer();

			cudaMemcpy(odata, odev, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy for odev failed");

			int numbool = 0;
			cudaMemcpy(&numbool, boolarr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

			int numindices = 0;
			cudaMemcpy(&numindices, indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

			int total = numbool + numindices;
			cudaFree(indices);
			cudaFree(idev);
			cudaFree(odev);
			cudaFree(boolarr);

			return total;
        }
    }
}
