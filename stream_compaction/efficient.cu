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

		__global__ void fillempty(int n,int * start, int fill)
		{
			int index= threadIdx.x;

			start[index+n] = fill;		
		}

        //Refered from Nvidia 
		//https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_pref01.html
		//
		__global__ void prescan(int *g_odata, int *g_idata, int n, int*temp)
		{
			//extern __shared__ int temp[];  // allocated on invocation
						
			int thid = blockIdx.x * blockDim.x + threadIdx.x;
			int offset = 1;

			temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
			temp[2 * thid + 1] = g_idata[2 * thid + 1];

			int d,ai,bi;
	
			// is >> faster than /=2 for int? 
			for (d = n >> 1; d > 0; d >>= 1)// build sum in place up the tree
			{		
				__syncthreads();
				if (thid < d)
				{
					ai = offset*(2 * thid + 1) - 1;
					bi = offset*(2 * thid + 2) - 1;

					temp[bi] += temp[ai];
				}
				offset *= 2;				
			}
			
			if (thid == 0) 
			{ 		
				temp[n - 1] = 0; // clear the last element
			} 
			
			int t;

			for (d = 1; d < n; d *= 2) // traverse down tree & build scan
			{
				offset >>= 1;
				__syncthreads();
				if (thid < d)
				{
					ai = offset*(2 * thid + 1) - 1;
					bi = offset*(2 * thid + 2) - 1;

					t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			__syncthreads();

			g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
			g_odata[2 * thid + 1] = temp[2 * thid + 1];		
		}

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
        void scan(int n, int *odata, const int *idata) {

			//Deal with non pow2 arrays
			int N;
			int count;			
			count = ilog2(n);
			N = 1<<count;
			if (N < n)
			{
				N *=2;
			}
			// Malloc buffers
			int* buffer3;
			int* buffer4;
			int* temp;
			cudaMalloc((void**)&buffer3, N * sizeof(int));
			cudaMalloc((void**)&buffer4, (N + 4) * sizeof(int));
			cudaMalloc((void**)&temp, (N + 4) * sizeof(int));
			cudaMemcpy(buffer3, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to buffer1 in GPU
			cudaMemcpy(buffer4, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to buffer2 in GPU
			cudaMemcpy(temp, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to buffer2 in GPU
			//Fill all empty positions as 0
			fillempty << <1, (N - n) >> > (n,buffer3, 0);
			fillempty << <1, (4 + N - n) >> > (n,buffer4, 0);
			fillempty << <1, (4 + N - n) >> > (n,temp, 0);

			//Start Timer
			timer().startGpuTimer();
            // TODO
			prescan<<<1,N/2>>>(buffer4, buffer3, N,temp);
			//END Timer
			timer().endGpuTimer();
           
			//COPY data back to CPU
			cudaMemcpy(odata, buffer4+1, N * sizeof(int), cudaMemcpyDeviceToHost);
			//Should I do this????
			odata[n - 1] = odata[n - 2] + idata[n - 1];
			//Free buffers
			cudaFree(buffer3);
			cudaFree(buffer4);
			cudaFree(temp);		
        }
       		


		__global__ void naive_sum2(int n, int* odata, int* idata)
		{
			int index = threadIdx.x;
			int i;
			int * ping;
			int * pong;
			int * swap;
			ping = idata;
			pong = odata;
			__syncthreads();

			for (i = 1; i < n; i *= 2)
			{
				if (index - i >= 0)
				{
					pong[index] = ping[index] + ping[index - i];
				}
				else
				{
					pong[index] = ping[index];
				}
				//Ping-Pong here!
				swap = ping;
				ping = pong;
				pong = swap;
				__syncthreads();
			}
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

			int* data;		
			cudaMalloc((void**)&data, n * sizeof(int));
			cudaMemcpy(data, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to data in GPU

			int* out;
			cudaMalloc((void**)&out, n * sizeof(int));
			cudaMemcpy(out, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to data in GPU

			int* bools;
			cudaMalloc((void**)&bools, n * sizeof(int));

			int* index;
			cudaMalloc((void**)&index, n * sizeof(int));


			timer().startGpuTimer();

			Common::kernMapToBoolean<<<1,n>>>(n, bools, data); //map the bools

			//TEST
			cudaMemcpy(odata, bools, n * sizeof(int), cudaMemcpyDeviceToHost); //COPY from idata in CPU to data in GPU
			int i;
			int count = 0;

			
			for (i = 0; i < n; i++)
			{
				count += odata[i];
				odata[i] = count-1;
			}
				
			cudaMemcpy(index, odata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to data in GPU
			cudaMemcpy(data, idata, n * sizeof(int), cudaMemcpyHostToDevice); //COPY from idata in CPU to data in GPU

			Common::kernScatter << <1, n>> > (n, out, data, bools, index);//write to out

			timer().endGpuTimer();

			//COPY data back to CPU
			cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost); //COPY back to CPU
			
			//Handle the first
			if(idata[0]!=0)
			{
				odata[0] = idata[0];
			}	
            return count;
        }
    }
}
