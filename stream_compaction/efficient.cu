#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 896

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		int *dev_x;
		int *dev_bools;
		int *dev_indices;
		int *dev_xcopy;
		__global__ void kernScan(int n, int *dev_x) {
			//=============================
			//====without shared memory====
			//=============================
			int t;
			int index = threadIdx.x;
			if (index >= n)
				return;
			int rIndex = n - 1 - index;
			int offset = 1;
			for (; offset < n; offset *= 2) {
				///only particular index need to do the operation.
				if ((index - offset >= 0) && (rIndex % (offset * 2) == 0))
					dev_x[index] += dev_x[index -offset];
				__syncthreads();
			}
			dev_x[n - 1] = 0;
			for (offset /= 2; offset > 0; offset /= 2) {
				if (index - offset >= 0 && (rIndex % (offset * 2) == 0)) {
					auto t = dev_x[index];
					dev_x[index] += dev_x[index - offset];
					dev_x[index - offset] = t;
				}
				__syncthreads();
			}

			//=============The following is a version with bug=============
			//======================Please ignore it=======================
			//Up-Sweep Phase
			/*
			for (int offset = 1; offset < n; offset *= 2) {
			///only particular index need to do the operation.
			if (index % (offset * 2) == 0)
			dev_x[index + offset * 2 - 1] += dev_x[index + offset - 1];
			__syncthreads();
			}
			//Down-Sweep Phase
			dev_x[n - 1] = 0;
			for (int offset = 1 << (ilog2ceil(n)); offset >= 1; offset /= 2) {
			///Same as above
			if (index % (offset * 2) == 0) {
			t = dev_x[index + offset - 1];
			dev_x[index + offset - 1] = dev_x[index + offset * 2 - 1];
			dev_x[index + offset * 2 - 1] += t;
			}
			__syncthreads();
			}
			*/
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_x, n * sizeof(int));
			cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			
			//timer().startGpuTimer();
            // TODO
			kernScan << < fullBlocksPerGrid, blockSize >> > (n, dev_x);
            //timer().endGpuTimer();

			cudaMemcpy(odata, dev_x, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_x);
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
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int cnt = 0;
			
			//count the size of output(# of non-zero element in idata)
			for (int i = 0; i < n; ++i) {
				if (idata[i] != 0)
					++cnt;
			}

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMalloc((void**)&dev_x, n * sizeof(int));
			cudaMalloc((void**)&dev_xcopy, n * sizeof(int));
			cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();
            // TODO
			//map
			StreamCompaction::Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_x);
			//scan
			cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			kernScan << < fullBlocksPerGrid, blockSize >> > (n, dev_indices);
			//scatter
			cudaMemcpy(dev_xcopy, dev_x, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			StreamCompaction::Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (n, dev_x, dev_xcopy, dev_bools, dev_indices);

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_x, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_x);
			cudaFree(dev_xcopy);

            return cnt;
        }
    }
}
