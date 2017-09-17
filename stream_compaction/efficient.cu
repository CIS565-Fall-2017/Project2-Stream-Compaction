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

		__global__ void upSweep(int n, int factorPlusOne, int factor, int *idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < n)
			{
				if (index % factorPlusOne == 0)
				{
					idata[index + factorPlusOne - 1] += idata[index + factor - 1];
				}
			}
		}//end upSweep function

		__global__ void downSweep(int n, int factorPlusOne, int factor, int *idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < n)
			{
				if (index % factorPlusOne == 0)
				{
					int leftChild = idata[index + factor - 1];
					idata[index + factor - 1] = idata[index + factorPlusOne - 1];
					idata[index + factorPlusOne - 1] += leftChild;
				}
			}
		}//end downSweep function

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.

		 * Notes:
			 Most of the text in Part 2 applies.
			 This uses the "Work-Efficient" algorithm from GPU Gems 3, Section 39.2.2.
			 This can be done in place - it doesn't suffer from the race conditions of the naive method, 
			 since there won't be a case where one thread writes to and another thread reads from the same location in the array.
			 Beware of errors in Example 39-2. Test non-power-of-two-sized arrays.
			 Since the work-efficient scan operates on a binary tree structure, it works best with arrays with power-of-two length. 
			 Make sure your implementation works on non-power-of-two sized arrays (see ilog2ceil). 
			 This requires extra memory, so your intermediate array sizes 
			 will need to be rounded to the next power of two.
         */
        void scan(int n, int *odata, const int *idata) {
			// TODO
			
			//If non-power-of-two sized array, round to next power of two
			int new_n;
			int *new_idata;

			if (n % 2 != 0)
			{
				new_n = 1 << ilog2ceil(n);
				new_idata = new int[new_n];
				
				//Fill the rest of the space with 0's
				for (int i = 0; i < new_n; i++)		//SHOULD THIS BE < or <=????
				{
					if (i < n)
					{
						new_idata[i] = idata[i];
					}
					else
					{
						new_idata[i] = 0;
					}
				}
			}
			else
			{
				new_n = n;
				new_idata = new int[new_n];
				cudaMemcpy(new_idata, idata, sizeof(int) * new_n, cudaMemcpyHostToHost);
			}
			
			dim3 fullBlocksPerGrid((new_n + blockSize - 1) / blockSize);

			int *inArray;
			cudaMalloc((void**)&inArray, new_n * sizeof(int));
			checkCUDAError("cudaMalloc inArray failed!");

			//Copy input data to GPU
			cudaMemcpy(inArray, new_idata, sizeof(int) * new_n, cudaMemcpyHostToDevice);
			cudaThreadSynchronize();
			
			bool timerHasStartedElsewhere = false;
			try
			{
				timer().startGpuTimer();
			}
			catch (std::runtime_error &e)
			{
				timerHasStartedElsewhere = true;
			}

			//Up sweep
			for (int d = 0; d <= ilog2ceil(new_n) - 1; d++)
			{
				int factorOut = 1 << (d + 1);	//2^(d + 1)
				int factorIn = 1 << d;			//2^d
				upSweep<<<fullBlocksPerGrid, blockSize>>>(new_n, factorOut, factorIn, inArray);

				//Make sure the GPU finishes before the next iteration of the loop
				cudaThreadSynchronize();
			}

			//Down sweep
			int lastElem = 0;
			cudaMemcpy(inArray + (new_n - 1), &lastElem, sizeof(int) * 1, cudaMemcpyHostToDevice);

			for (int d = ilog2ceil(new_n) - 1; d >= 0; d--)
			{
				int factorPlusOne = 1 << (d + 1);	//2^(d + 1)
				int factor = 1 << d;				//2^d
				downSweep<<<fullBlocksPerGrid, blockSize>>>(new_n, factorPlusOne, factor, inArray);
				cudaThreadSynchronize();
			}

			if (!timerHasStartedElsewhere)
			{
				timer().endGpuTimer();
			}

			//Transfer to odata
			cudaMemcpy(odata, inArray, sizeof(int) * (new_n), cudaMemcpyDeviceToHost);

			//Free the temp device array
			cudaFree(inArray);
        }//end scan function 

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
			
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			int *inArray;
			int *boolsArray;
			cudaMalloc((void**)&inArray, n * sizeof(int));
			checkCUDAError("cudaMalloc inArray failed!");
			cudaMalloc((void**)&boolsArray, n * sizeof(int));
			checkCUDAError("cudaMalloc boolsArray failed!");

			int* scan_in = new int[n];
			int* scan_out = new int[n];

			int *scatter_in;
			int *scatter_out;
			cudaMalloc((void**)&scatter_in, n * sizeof(int));
			checkCUDAError("cudaMalloc scatter_in failed!");
			cudaMalloc((void**)&scatter_out, n * sizeof(int));
			checkCUDAError("cudaMalloc scatter_out failed!");

			cudaThreadSynchronize();

			//Copy input data to GPU
			cudaMemcpy(inArray, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			//Call kernMapToBoolean to map values to bool array
			Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, boolsArray, inArray);

			//Copy back to host array, find how many fulfilled condition, and run exclusive scan
			cudaMemcpy(scan_in, boolsArray, sizeof(int) * n, cudaMemcpyDeviceToHost);

			int numPassedElements = 0;
			for (int i = 0; i < n; i++)
			{
				if (scan_in[i] == 1)
				{
					numPassedElements++;
				}
			}

			scan(n, scan_out, scan_in);

			//Copy output of CPU scan to scatter device array
			cudaMemcpy(scatter_in, scan_out, sizeof(int) * n, cudaMemcpyHostToDevice);
			
			//Call kernScatter with scanned boolsArray
			Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, scatter_out, inArray, boolsArray, scatter_in);

			timer().endGpuTimer();

			//SCATTER OUT ISNT GONNA BE THE SAME SIZE AS N
			//Should I replace n with numPassedElements? 
			cudaMemcpy(odata, scatter_out, sizeof(int) * n, cudaMemcpyDeviceToHost);

			//Free the device arrays
			cudaFree(inArray);
			cudaFree(boolsArray);
			cudaFree(scatter_in);
			cudaFree(scatter_out);

            return numPassedElements;

        }//end compact function
    }//end namespace Efficient
}//end namespace StreamCompaction
