#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <vector>
static const int blockSize{ 256};

void printArray2(int n, int *a, bool abridged = false) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15 && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%3d ", a[i]);
	}
	printf("]\n");
}



namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

	// initialize the arrays on the GPU
	// returns the addresses of the pointers on the GPU
	// dev_idata is a pointer to the address of the dev_idata array that
	// gets updated here
	// initialize dev_idata has the first
        // elements copied and the remainder to make the stream 2^n
        // are set to 0. The first input is the size of the arrays
        // to allocate and the second input is the size of the array to transfer.
	// N the maximum size of the allocated array.  n is the size of the data array
	// N is one more than the multiple of 2 greater or equal to n, 
	// in dev_idata, and then the elements are copied inte dev_idata.
	
	void initScan(int N, int n, const int *idata, int ** dev_idata)
        {
            int size {sizeof(int)};
            cudaMalloc((void**) (dev_idata), N * size);
            checkCUDAError("Allocating Scan Buffer Efficient Error"); 
	        cudaMemset(*dev_idata, 0,  N * size);
	        cudaMemcpy(*dev_idata , idata, n *size, cudaMemcpyHostToDevice);
	        // no need to initialize the odata because the loop does that each time
	        checkCUDAError("Initialize and Copy data to target Error");
	        cudaThreadSynchronize();
	}
        // transfer scan data back to host
	void transferScan(int N, int * odata, int * dev_odata)
	{
		cudaMemcpy(odata, dev_odata, N * sizeof(int), cudaMemcpyDeviceToHost);
	}
		
	// end the scan on the device.
	void endScan( int * dev_idata)
	{
		cudaFree(dev_idata);
	}
	// kernParallelReduction uses contiguous threads to do the parallel reduction
	// There is one thread for every two elements
	__global__ void kernParallelReduction(int N, int Stride, 
		int iteration, int * dev_idata)
	{
		int thread = threadIdx.x + blockIdx.x * blockDim.x;
		if (thread >= N >> iteration ) {
			return;				
		}
		int priorStride { Stride >> 1};
		int index = (thread + 1) * Stride  -1;
		if (index < N) {
			    dev_idata[index] += dev_idata[index - priorStride];
		}
	}
	// Downsweep uses contiguous threads to sweep down and add the intermediate
	// results to the partial sums already computed
	// There is one thread for every two elements.  Here there is a for loop 
	// that changes the stride.  Contiguous allows the first threads to do all
	// the work and later warps will all be 0.
	__global__ void kernDownSweep(int N, int stride, int iteration, int * dev_idata)
	{
		int thread = threadIdx.x + blockIdx.x * blockDim.x;
		if ( thread >= N >> iteration) {
			return;
		}
		// have one thread set the last element to 0;
		int startOffset { N - 1};
		int right = - stride * thread + startOffset;
		if (right >= 0) {
		   int separation = stride >> 1;
		   int left = right - separation;
		   int current = dev_idata[right];
		   dev_idata[right] += dev_idata[left];
		   dev_idata[left]   = current;
		}
	}
//	__global__ void kernDownSweep(int N, int * dev_idata)
//	{
//		int thread = threadIdx.x + blockIdx.x * blockDim.x;
//		if ( 2 * thread >= N) {
//			return;
//		}
//		// have one thread set the last element to 0;
//		int startOffset { N - 1};
//		if (thread == 0) {
//			dev_idata[startOffset] = 0;
//		}
//		for ( int stride = {N}; stride > 1; stride >>= 1)
//		{
//			int right = - stride * thread + startOffset;
//			if (right >= 0) {
//			   int separation = stride >> 1;
//			   int left = right - separation;
//			   int current = dev_idata[right];
//			   dev_idata[right] += dev_idata[left];
//			   dev_idata[left]   = current;
//			}
//			__syncthreads();
//		}
//	}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
	void efficientScan(int N,  int d, int * dev_idata)
	{
            int Nthreads {N>>1};
            dim3  fullBlocksPerGrid((Nthreads + blockSize - 1)/ blockSize);
			  int iteration{ 1 };
	          for ( int stride = {2}; stride <= N; stride *= 2)
	          {
                    kernParallelReduction<<<fullBlocksPerGrid, blockSize>>>
		            (N, stride, iteration, dev_idata);
					++iteration;
	          }
	          cudaMemset((dev_idata + N - 1), 0, sizeof(int));
			  iteration = d;
	          for ( int stride = {N}; stride > 1; stride >>= 1)
	          {
	             kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(N,  stride,
					 iteration, dev_idata);
				 --iteration;
	          }
	}
        void scan(int n, int *odata, const int *idata) 
	{
            int * dev_idata;
            // d is the number of scans needed and also the
            // upper bound for log2 of the number of elements
            int d {ilog2ceil(n)}; //
            int N { 1 << d };
            initScan(N, n, idata, &dev_idata);	
            timer().startGpuTimer();
            efficientScan( N, d, dev_idata);
            timer().endGpuTimer();
            // only transfer tho first n elements of the 
            // exclusive scan
            transferScan(n, odata, dev_idata);
            endScan(dev_idata);
        }

	void initCompact(int N, int n, const int *idata, int ** dev_idata, 
			int **dev_booldata, int  ** dev_indices, int **dev_odata)
        {
            int size {sizeof(int)};
            cudaMalloc(reinterpret_cast<void**> (dev_booldata), N * size);
            cudaMalloc(reinterpret_cast<void**> (dev_idata), N * size);
            cudaMalloc(reinterpret_cast<void**> (dev_indices), N * size);
            cudaMalloc(reinterpret_cast<void**> (dev_odata), N * size);
            checkCUDAError("Allocating Compaction Scan Error"); 
            cudaMemset(*dev_idata, 0,  N * size);
	    cudaMemcpy(*dev_idata , idata, n *size, cudaMemcpyHostToDevice);
	    // no need to initialize the odata because the loop does that each time
            checkCUDAError("Initialize and Copy data to target Error");
	    cudaThreadSynchronize();
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

        int * dev_idata;
	    int * dev_booldata;
	    int * dev_indices;
	    int * dev_odata;
            // d is the number of scans needed and also the
            // upper bound for log2 of the number of elements
            int d {ilog2ceil(n)}; //
            int N { 1 << d };
            initCompact(N, n, idata, &dev_idata, &dev_booldata, &dev_indices,
			    &dev_odata);	
            timer().startGpuTimer();
            dim3  fullBlocksPerGrid((N + blockSize - 1)/ blockSize);
            
	    StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid,
		         blockSize>>>(N, dev_booldata, dev_idata);
	    cudaMemcpy(dev_indices, dev_booldata, N * sizeof(int), 
			    cudaMemcpyDeviceToDevice);
            efficientScan( N, d, dev_indices);
	    StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid,
		             blockSize>>>(N, dev_odata, dev_idata, 
					  dev_booldata, dev_indices);
	    timer().endGpuTimer();
            int  lastIndex;
	    transferScan(1, &lastIndex, dev_indices + N - 1);
	    int lastIncluded;
	    transferScan(1, &lastIncluded, dev_booldata + N - 1);
		std::vector<int> input(n);
		std::vector<int> bools(n);
		std::vector<int> indices(n);
		transferScan(n, input.data(), dev_idata);
		transferScan(n, bools.data(), dev_booldata);
		transferScan(n, indices.data(), dev_indices);
		printArray2(n, input.data(), true);
		printArray2(n, bools.data(), true);
		printArray2(n, indices.data(), true);
	    n = lastIncluded + lastIndex;
	    transferScan(n, odata, dev_odata);
		printArray2(n, odata, true);
		endScan(dev_odata);
		endScan(dev_idata);
		endScan(dev_indices);
		endScan(dev_booldata);
        return n;
        }
    }
}
