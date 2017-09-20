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

		__global__ void kern_upSweep(int n, int d, int* idata) {
			int index = (threadIdx.x + (blockIdx.x * blockDim.x));
			int k = index * (1 << d + 1);
			if (index >= n || k >= n) { return; }

			idata[k + (1 << d+1) - 1] += idata[k + (1 << d) - 1];
		}

		__global__ void kern_downSweep(int n, int d, int* idata) {
			int k = (threadIdx.x + (blockIdx.x * blockDim.x)) * (1 << d + 1);
			if (k >= n) { return; }

			int t = idata[k + (1 << d) - 1];
			idata[k + (1 << d)   - 1] = idata[k + (1 << d+1) - 1];
			idata[k + (1 << d+1) - 1] += t;
		}

		__global__ void roundN(int n, int nRounded, int* idataRounded, const int* idata) {
			int i = (threadIdx.x + (blockIdx.x * blockDim.x));
			if (i >= nRounded) { return; }

			idataRounded[i] = i >= n ? 0 : idata[i];
		}

		__global__ void kern_scan_shared(int n, int *odata, const int *idata) {

			extern __shared__ float temp[];


			
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*
		***** THIS IS AN EFFICIENT VERSION USING SHARED MEMORY
		*/
		void scan_shared(int n, int *odata, const int *idata) {
			// Super Hyperthreaded Information Transloading calculation for threads per block
			dim3 threadsPerBlock(std::min(getThreadsPerBlock(), n));
			dim3 numBlocks(std::ceilf(((float)n / threadsPerBlock.x)));


		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// Super Hyperthreaded Information Transloading calculation for threads per block
			dim3 threadsPerBlock(std::min(getThreadsPerBlock(), n));
			dim3 numBlocks(std::ceilf(((float)n / threadsPerBlock.x)));

			//Round Up
			int loops = ilog2ceil(n);
			int nRounded = 1 << loops;

			// A copy of idata on the GPU
			int* idata_GPU;
			cudaMalloc((void**)&idata_GPU, sizeof(int) * n);
			cudaMemcpy(idata_GPU, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			try { timer().startGpuTimer(); }
			catch (...) {};

			//Rounded Version of GPU Copy
			int* idataRounded_GPU;
			cudaMalloc((void**)&idataRounded_GPU, sizeof(int) * nRounded);
			//Round the GPU Array
			roundN << <numBlocks, threadsPerBlock >> > (n, nRounded, idataRounded_GPU, idata_GPU);

			//Up-Sweep:
			for (int d = 0; d < loops; d++) {
				kern_upSweep<< <numBlocks, threadsPerBlock >> >(n, d, idataRounded_GPU);
				checkCUDAErrorFn("upSweep failed with error");
			}

			//Set Zero
			int zero = 0;
			cudaMemcpy(&idataRounded_GPU[nRounded - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Zero Copy failed with error");

			//Down-Sweep:
			for (int d = loops - 1; d >= 0; d--) {
				kern_downSweep <<<numBlocks, threadsPerBlock >> >(nRounded, d, idataRounded_GPU);
				checkCUDAErrorFn("downSweep failed with error");
			}

			cudaMemcpy(odata, idataRounded_GPU, sizeof(int) * n, cudaMemcpyDeviceToHost);

			//Free Malloc'd
			cudaFree(idataRounded_GPU);
			cudaFree(idata_GPU);
			/**** PRINTER ******
			printf("After DownSweep: \n (");
			for (int i = nRounded-10; i < nRounded -1; i++) {
				printf("%d = %d, ", i, odata[i]);
			}
			printf("%d = %d) \n\n", nRounded-1, odata[nRounded-1]);
			**/
			try { timer().endGpuTimer(); }
			catch (...) {};
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
			try { timer().startGpuTimer(); }
			catch (...) {};
			// Super Hyperthreaded Information Transloading calculation for threads per block
			dim3 threadsPerBlock(std::min(getThreadsPerBlock(), n));
			dim3 numBlocks(std::ceilf(((float)n / threadsPerBlock.x)));

            // Create Buffers
			int *temp, *scanned, *idata_GPU, *odata_GPU, *count_GPU;
			cudaMalloc((void**)&temp     , sizeof(int) * n);
			cudaMalloc((void**)&scanned  , sizeof(int) * n);
			cudaMalloc((void**)&idata_GPU, sizeof(int) * n);
			cudaMalloc((void**)&odata_GPU, sizeof(int) * n);

			cudaMemcpy(idata_GPU, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("idata memcpy failed with error");
			
			//Create Temp Array
			Common::kernMapToBoolean << <numBlocks, threadsPerBlock >> > (n, temp, idata_GPU);
			checkCUDAErrorFn("kern_boolify failed with error");

			//Scan
			scan(n, scanned, temp);

			//Temporarily store "scanned" into odata to get count
			cudaMemcpy(odata, scanned, sizeof(int) * n, cudaMemcpyDeviceToHost);
			int count = odata[n - 1] + (int)(idata[n - 1] != 0);
			
			//Compact
			Common::kernScatter << <numBlocks, threadsPerBlock >> >(n, odata_GPU, idata_GPU, temp, scanned);
			checkCUDAErrorFn("kern_compact failed with error");

			//Bring Back to CPU
			cudaMemcpy(odata,  odata_GPU, sizeof(int) * count, cudaMemcpyDeviceToHost);
			
			//Free Up All Malloc'd
			cudaFree(temp);
			cudaFree(scanned);
			cudaFree(idata_GPU);
			cudaFree(odata_GPU);


			try { timer().endGpuTimer(); }
			catch (...) {};
            return count;
        }
    }
}
