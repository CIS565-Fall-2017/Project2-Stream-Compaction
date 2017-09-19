#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "cpu.h"
#include "naive.h"

#define BLOCKSIZE 512
#define SOLVE_BANK_CONFLICTS 1

#if BLOCKSIZE > 1024
#error "Warning: Blocksize cannot excess 1024!"
#endif

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

typedef int var_t;
int scan_timing = 1;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#pragma region PrescanEfficientWithShareMemory
		__global__ void scanWithShareMem(int n, int * idata, int * odata, int * aux) {
			
			// Declare Share Memory
			#if  SOLVE_BANK_CONFLICTS
				__shared__ var_t temp[BLOCKSIZE + NUM_BANKS];
			#else
				__shared__ var_t temp[BLOCKSIZE];
			#endif

			// Declare necessary variables
			const var_t tid = threadIdx.x;
			const var_t bid = blockIdx.x;
			const var_t s = bid * BLOCKSIZE;
			var_t offset = 1;

			// Copy Global Memory to Share Memory
			#if  SOLVE_BANK_CONFLICTS
				var_t ai = tid << 1;
				var_t bi = ai + 1;
				var_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
				var_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
				temp[ai + bankOffsetA] = idata[ai + s];
				temp[bi + bankOffsetB] = idata[bi + s];
			#else
				temp[2 * tid] = idata[2 * tid + s];
				temp[2 * tid + 1] = idata[2 * tid + s + 1];
			#endif

			// Reduction Phase
			for (var_t d = BLOCKSIZE >> 1; d > 0; d >>= 1)
			{
				__syncthreads();
				if (tid < d)
				{
					var_t ai = offset*((tid << 1) + 1) - 1;
					var_t bi = offset*((tid << 1) + 2) - 1;
					#if  SOLVE_BANK_CONFLICTS
						ai += CONFLICT_FREE_OFFSET(ai);
						bi += CONFLICT_FREE_OFFSET(bi);
					#endif
					temp[bi] += temp[ai];
				}
				offset <<= 1;
			}

			// Copy Block Sum to Aux Array
			__syncthreads();
			if (!tid) {
				#if  SOLVE_BANK_CONFLICTS
					aux[bid] = temp[BLOCKSIZE - 1 + CONFLICT_FREE_OFFSET(BLOCKSIZE - 1)];
					temp[BLOCKSIZE - 1 + CONFLICT_FREE_OFFSET(BLOCKSIZE - 1)] = 0;
				#else
					aux[bid] = temp[BLOCKSIZE - 1];
					temp[BLOCKSIZE - 1] = 0;
				#endif
			}

			for (var_t d = 1; d < BLOCKSIZE; d <<= 1)
			{
				offset >>= 1;
				__syncthreads();
				if (tid < d)
				{
					int ai = offset*((tid << 1) + 1) - 1;
					int bi = offset*((tid << 1) + 2) - 1;
					#if  SOLVE_BANK_CONFLICTS
						ai += CONFLICT_FREE_OFFSET(ai);
						bi += CONFLICT_FREE_OFFSET(bi);
					#endif
					var_t t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}

			__syncthreads();
		#if  SOLVE_BANK_CONFLICTS
			 ai = tid << 1;
			 bi = ai + 1;
			 bankOffsetA = CONFLICT_FREE_OFFSET(ai);
			 bankOffsetB = CONFLICT_FREE_OFFSET(bi);
			odata[ai + s] = temp[ai + bankOffsetA];
			odata[bi + s] = temp[bi + bankOffsetB];
		#else
			odata[2 * tid + s] = temp[2 * tid]; 
			odata[2 * tid + 1 + s] = temp[2 * tid + 1];
		#endif
		}

		__global__ void sumWithBlock(int n, int * idata, int * odata, int * aux) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			int bid = blockIdx.x;
			if (idx >= n) {
				return;
			}
			
			if(bid > 0) {
				odata[idx] = idata[idx] + aux[bid];
			}
			else {
				odata[idx] = idata[idx];
			}
		}

		void scan_s(int n, int *odata, int *idata) {
			int dSize, dLen, _n, aSize, aLen;
			int *dev_idata, *dev_odata;
			int *host_aux, *host_aux_sum;
			int *dev_aux, *dev_aux_sum;

			_n = ilog2ceil(n);
			dLen = 1 << _n;
			dSize = dLen * sizeof(int);

			aLen = ((n + BLOCKSIZE - 1) / BLOCKSIZE);
			aSize = aLen * sizeof(int);

			dim3 blocksPerGrid((n + BLOCKSIZE - 1) / (BLOCKSIZE));
			dim3 threadsPerBlocks(BLOCKSIZE / 2);
			dim3 blocksPerGrid_aux((aLen + BLOCKSIZE - 1) / (BLOCKSIZE));
			dim3 threadsPerBlocks_aux(BLOCKSIZE / 2);

			// Alloc variables
			cudaMalloc((void**)&dev_odata, dSize);
			cudaMalloc((void**)&dev_idata, dSize);
			cudaMalloc((void**)&dev_aux, aSize);
			cudaMalloc((void**)&dev_aux_sum, aSize);
			cudaMallocHost((void**)&host_aux, aSize); // Use Pin-Memory to ensure the maximum memory speed
			cudaMallocHost((void**)&host_aux_sum, aSize);

			cudaMemcpy(dev_idata, idata, dSize, cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			// Prescan Each Block use GPU
			scanWithShareMem << <blocksPerGrid, threadsPerBlocks >> >(n, dev_idata, dev_odata, dev_aux);

			// Prescan Auxiliary Array use CPU
			cudaMemcpy(host_aux, dev_aux, aSize, cudaMemcpyDeviceToHost);
			StreamCompaction::CPU::scan_incusive(aLen, host_aux_sum, host_aux);

			// Sum up with Blocks use GPU
			cudaMemcpy(dev_aux_sum, host_aux_sum, aSize, cudaMemcpyHostToDevice);
			sumWithBlock << <blocksPerGrid, BLOCKSIZE >> >(n, dev_odata, dev_idata, dev_aux_sum);

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_aux);
			cudaFree(dev_aux_sum);
			cudaFreeHost(host_aux);
			cudaFreeHost(host_aux_sum);
		}

#pragma endregion


#pragma region PrescanEfficientWithOptimization
		__global__ void reduction(const int _d, const int ts, int * idata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= ts) {
				return;
			}

			int ai = idx * _d + _d - 1;
			int bi = ai - (_d >> 1);

			idata[ai] += idata[bi];
		}

		__global__ void downSweep(const int _d, const int ts, int * idata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= ts) {
				return;
			}

			int ai = idx * _d + _d - 1;
			int bi = ai - (_d >> 1);

			int t = idata[bi];
			idata[bi] = idata[ai];
			idata[ai] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			
			int dSize, dLen, _n;
			int *dev_data;

			dim3 threadsPerBlocks(BLOCKSIZE);

			_n = ilog2ceil(n);
			dLen = 1 << _n;
			dSize = dLen * sizeof(int);
			
			cudaMalloc((void**)&dev_data, dSize);
			cudaMemcpy(dev_data, idata, dSize, cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			int _d, ts;
			for (int d = 0; d < _n; d++) {
				ts = 1 << (_n - d - 1);
				_d = 1 << (d + 1);

				dim3 blocksPerGrid((ts + BLOCKSIZE - 1) / BLOCKSIZE);
				reduction<<<blocksPerGrid, threadsPerBlocks>>>(_d, ts, dev_data);
			}
			
			cudaMemset(dev_data + dLen - 1, 0, sizeof(int));
			
			for (int d = _n - 1; d > -1; d--) {
				ts = 1 << (_n - d - 1);
				_d = 1 << (d + 1);

				dim3 blocksPerGrid((ts + BLOCKSIZE - 1) / BLOCKSIZE);
				downSweep<<<blocksPerGrid, threadsPerBlocks>>>(_d, ts, dev_data);
			}
			
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			
			cudaFree(dev_data);
        }

		void scan_dev(int len, int _n, int *data) {
			int _d, ts;
			dim3 threadsPerBlocks(BLOCKSIZE);

			for (int d = 0; d < _n; d++) {
				ts = 1 << (_n - d - 1);
				_d = 1 << (d + 1);

				dim3 blocksPerGrid((ts + BLOCKSIZE - 1) / BLOCKSIZE);
				reduction << <blocksPerGrid, threadsPerBlocks >> >(_d, ts, data);
			}

			cudaMemset(data + len - 1, 0, sizeof(int));

			for (int d = _n - 1; d > -1; d--) {
				ts = 1 << (_n - d - 1);
				_d = 1 << (d + 1);

				dim3 blocksPerGrid((ts + BLOCKSIZE - 1) / BLOCKSIZE);
				downSweep << <blocksPerGrid, threadsPerBlocks >> >(_d, ts, data);
			}
		}

#pragma endregion


#pragma region PrescanEfficient
		__global__ void reduction0(const int _d, const int n, int * idata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n || (idx % _d)) {
				return;
			}

			idata[idx + _d - 1] += idata[idx + (_d >> 1) - 1];
		}

		__global__ void downSweep0(const int _d, const int n, int * idata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n || (idx % _d)) {
				return;
			}

			int t = idata[idx + (_d >> 1) - 1];
			idata[idx + (_d >> 1) - 1] = idata[idx + _d - 1];
			idata[idx + _d - 1] += t;
		}

		void scan0(int n, int *odata, const int *idata) {

			int dSize, dLen, _n;
			int *dev_data;

			dim3 threadsPerBlocks(BLOCKSIZE);

			_n = ilog2ceil(n);
			dLen = 1 << _n;
			dSize = dLen * sizeof(int);

			cudaMalloc((void**)&dev_data, dSize);
			cudaMemcpy(dev_data, idata, dSize, cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			int _d;
			for (int d = 0; d < _n; d++) {
				_d = 1 << (d + 1);

				dim3 blocksPerGrid((dLen + BLOCKSIZE - 1) / BLOCKSIZE);
				reduction0 << <blocksPerGrid, threadsPerBlocks >> >(_d, n, dev_data);
			}

			cudaMemset(dev_data + dLen - 1, 0, sizeof(int));

			for (int d = _n - 1; d > -1; d--) {
				_d = 1 << (d + 1);

				dim3 blocksPerGrid((dLen + BLOCKSIZE - 1) / BLOCKSIZE);
				downSweep0 << <blocksPerGrid, threadsPerBlocks >> >(_d, n, dev_data);
			}

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_data);
		}

#pragma endregion


#pragma region Compaction

        int compact(int n, int *odata, const int *idata) {
            
			int _n = ilog2ceil(n);
			int len = 1 << _n;
			int bsize = len * sizeof(int);
			int nsize =  n * sizeof(int);

			int *dev_in, *dev_out, *dev_bools, *dev_indices;

			cudaMalloc((void**)&dev_in, nsize);
			cudaMalloc((void**)&dev_out, nsize);
			cudaMalloc((void**)&dev_bools, bsize);
			cudaMalloc((void**)&dev_indices, bsize);

			cudaMemcpy(dev_in, idata, nsize, cudaMemcpyHostToDevice);

			cudaMemset(dev_bools, 0, bsize);

			dim3 blocksPerGrid((len + BLOCKSIZE - 1) / (BLOCKSIZE));
			dim3 threadsPerBlocks(BLOCKSIZE);

			scan_timing = 0;
			timer().startGpuTimer();

			// 1
			Common::kernMapToBoolean << <blocksPerGrid, threadsPerBlocks >> > (n, dev_bools, dev_in);
			
			// 2
			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
			scan_dev(len, _n, dev_indices);
			
			// 3
			Common::kernScatter << <blocksPerGrid, threadsPerBlocks >> >(n, dev_out, dev_in, dev_bools, dev_indices);

            timer().endGpuTimer();
			scan_timing = 1;

			
			int *test;
			test = (int *)malloc(n * sizeof(int));
			cudaMemcpy(test, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i < n; i++) {
				//printf("%i\n", test[i]);
			}
			
			// 4
			int s;
			cudaMemcpy(&s, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			s = idata[n - 1] ? s + 1 : s;

			cudaMemcpy(odata, dev_out, s * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			
            return s;
        }
		
#pragma endregion

    }
}
