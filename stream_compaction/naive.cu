#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
	#define blockSize 32
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernelScan(int *odata, int *idata, int n, int d)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int stride = 1 << (d - 1);
			if (index < n) {
				odata[index] = (index >= stride) ? idata[index - stride] + idata[index] : idata[index];
			}
		}

		void scan(int n, int *odata, const int *idata) {

			//GPU prep
			int *dev_odata, *dev_idata;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			dim3 threadsPerBlock(blockSize);


			timer().startGpuTimer();
			//actual kernel invocation
			int maxStride = ilog2ceil(n);
			for (int i = 1; i <= maxStride; i++) {
				kernelScan <<<fullBlocksPerGrid, blockSize >>> (dev_odata, dev_idata, n, i);
				int *temp = dev_idata;
				dev_idata = dev_odata;
				dev_odata = temp;
			}

			//this will end with idata holding the info, convert to exclusive
			Common::kernInclusiveToExclusive <<<fullBlocksPerGrid, blockSize >>> (n, dev_odata, dev_idata);

			timer().endGpuTimer();

			//send data back to cpu memory
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			//delete GPU arrays
			cudaFree(dev_odata);
			cudaFree(dev_idata);
		}
    }
}
