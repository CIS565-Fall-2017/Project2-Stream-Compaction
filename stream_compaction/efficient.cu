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
					idata[index + factorPlusOne - 1] += idata[index + factor - 1] ;
				}
			}
		}//end upSweep function

		__global__ void downSweep(int n, int factorPlusOne, int factor, int *idata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < n)
			{
				//if (index == 0)
				//{
				//	idata[n - 1] = 0;
				//}

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
            timer().startGpuTimer();
            
			// TODO

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			int *inArray;	//for up sweep
			cudaMalloc((void**)&inArray, n * sizeof(int));
			checkCUDAError("cudaMalloc inArray failed!");

			int *inArray2;	//for down sweep
			cudaMalloc((void**)&inArray2, n * sizeof(int));
			checkCUDAError("cudaMalloc inArray2 failed!");

			cudaThreadSynchronize();

			//Copy input data to GPU
			cudaMemcpy(inArray, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			//ADD SOME CONDITION TO MAKE SURE INARRAY IS ALWAYS SIZE OF POWER OF TWO

			//Up sweep
			for (int d = 0; d <= ilog2ceil(n) - 1; d++)
			{
				int factorOut = 1 << (d + 1);	//2^(d + 1)
				int factorIn = 1 << d;			//2^d
				upSweep<<<fullBlocksPerGrid, blockSize>>>(n, factorOut, factorIn, inArray);

				//Make sure the GPU finishes before the next iteration of the loop
				cudaThreadSynchronize();
			}

			//Down sweep

			//Can you not send CPU arrays into a kernel? Yeah probs not, sounds stupid
			//Is there a better way to do this? Should I just make a kernel of one block???

			//odata[n - 1] = 0;
			//cudaMemcpy(odata, inArray, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
			//cudaMemcpy(inArray2, odata, sizeof(int) * (n), cudaMemcpyHostToDevice);

			//Previous way is stupid with 2 memcpy's. Expensive to bring back to host. Do this instead.
			int lastElem = 0;
			cudaMemcpy(inArray + (n - 1), &lastElem, sizeof(int) * 1, cudaMemcpyHostToDevice);

			for (int d = ilog2ceil(n) - 1; d >= 0; d--)
			{
				int factorPlusOne = 1 << (d + 1);	//2^(d + 1)
				int factor = 1 << d;				//2^d

				downSweep<<<fullBlocksPerGrid, blockSize>>>(n, factorPlusOne, factor, inArray);
				//downSweep<<<fullBlocksPerGrid, blockSize>>>(n, factorPlusOne, factor, inArray2);

				cudaThreadSynchronize();
			}




			//Transfer to odata
			cudaMemcpy(odata, inArray, sizeof(int) * (n), cudaMemcpyDeviceToHost);
			//cudaMemcpy(odata, inArray2, sizeof(int) * (n), cudaMemcpyDeviceToHost);


			//Free the temp device array
			cudaFree(inArray);
			cudaFree(inArray2);

            timer().endGpuTimer();
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
            timer().endGpuTimer();
            return -1;
        }
    }
}
