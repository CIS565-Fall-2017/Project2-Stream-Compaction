/*
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "stream_compaction/common.h"
#include "stream_compaction/efficient.h"
#include "stream_compaction/radix.h"

#define BLOCKSIZE 128

typedef int var_t;

namespace StreamCompaction {
	namespace Radix {

		template<typename T>
		T findMax(T *arr, T n) {
			T max = 0;
			for (int i = 0; i < n; ++i) {
				if (arr[i] > max) {
					max = arr[i];
				}
			}
			return max;
		}

		__global__ void getB(const int n, const int t, int *idata, int *odata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}

			odata[idx] = ((idata[idx] & (1 << t)) ^ (1 << t));
		}

		__global__ void getE(const int n, int *idata, int *edata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}

			edata[idx] = (idata[idx] ^ 1);
		}

		__global__ void getTF(const int n, int *edata, int *fdata, int *odata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}

			odata[idx] = edata[idx] + fdata[idx];
		}

		__global__ void getT(const int n, const int tf, int *fdata, int *odata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}

			odata[idx] = idx - fdata[idx] + tf;
		}

		__global__ void getD(const int n, int *bdata, int * tdata, int * fdata, int *odata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}

			odata[idx] = (bdata[idx] ? tdata[idx] : fdata[idx]);
		}

		__global__ void refill(const int n, int * d, int * idata, int *odata) {
			int idx = threadIdx.x + (blockIdx.x * blockDim.x);
			if (idx >= n) {
				return;
			}

			odata[d[idx]] = idata[idx];
		}

		void sort(int n, int *odata, int *idata) {
			
			int size;
			int *dev_idata, *dev_odata;
			int *dev_b, *dev_e, *dev_f, *dev_t, *dev_d;

			size = n * sizeof(int);

			cudaMalloc((void**)&dev_idata, size);
			cudaMalloc((void**)&dev_odata, size);
			cudaMalloc((void**)&dev_b, size);
			cudaMalloc((void**)&dev_e, size);
			cudaMalloc((void**)&dev_f, size);
			cudaMalloc((void**)&dev_t, size);
			cudaMalloc((void**)&dev_d, size);

			dim3 blocksPerGrid((n + BLOCKSIZE - 1) / BLOCKSIZE);
			dim3 threadsPerBlock(BLOCKSIZE);

			var_t max = findMax<var_t>(idata, n);
			int ndigit = ilog2ceil(max);

			for (int i = 0; i < ndigit; i++) {
				getB<<<blocksPerGrid, threadsPerBlock>>>(n, i, dev_b, dev_idata);
			
				getE<<<blocksPerGrid, threadsPerBlock>>>(n, dev_b, dev_e);
			
				thrust::device_ptr<int> dev_thrust_e(dev_e);
				thrust::device_ptr<int> dev_thrust_f(dev_f);
				thrust::exclusive_scan(dev_thrust_e, dev_thrust_e + n, dev_thrust_f);

				int tf, le;
				cudaMemcpy(&tf, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(&le, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				tf += le;

				getT << <blocksPerGrid, threadsPerBlock >> >(n, tf, dev_f, dev_t);
			
				getD << <blocksPerGrid, threadsPerBlock >> >(n, dev_b, dev_t, dev_f, dev_d);
			
				refill(n, dev_d, dev_idata, dev_odata);
				std::swap(dev_idata, dev_odata);
			}

			cudaMemcpy(odata, dev_idata, size, cudaMemcpyDeviceToHost);

		}

	}
}*/