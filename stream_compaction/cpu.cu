#include <cstdio>
#include "cpu.h"
#include<iostream>
#include "common.h"

namespace StreamCompaction {
	namespace CPU {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		/**
		* CPU scan (prefix sum).
		* For performance analysis, this is supposed to be a simple for loop.
		* (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
		*/
		void scan(int n, int *odata, const int *idata) {

			if (n <= 0) return;
			memcpy(odata, idata, n * sizeof(int));
			int layer = ilog2ceil(n);
			int oLength = 1 << layer;

			// Uncomment the timer here if you want to test the efficiency of scan function
			timer().startCpuTimer();
			for (int d = 0; d < layer; d++) {
				for (int k = 0; k < oLength; k += (1 << (d + 1))) {

					odata[k + (1 << (d + 1)) - 1] += odata[k + (1 << d) - 1];
				}
			}
			odata[oLength - 1] = 0;
			for (int d = layer - 1; d >= 0; d--) {
				for (int k = 0; k < oLength; k += (1 << (d + 1))) {
					int nodeIdx = k + (1 << d) - 1;
					int temp = odata[nodeIdx];
					odata[nodeIdx] = odata[nodeIdx + (1 << d)];
					odata[nodeIdx + (1 << d)] += temp;
				}
			}
			timer().endCpuTimer();
		}

		/**
		* CPU stream compaction without using the scan function.
		*
		* @returns the number of elements remaining after compaction.
		*/
		int compactWithoutScan(int n, int *odata, const int *idata) {

			// TODO
			if (n <= 0) return -1;
			int num = 0;
			timer().startCpuTimer();
			for (int i = 0; i < n; i++) {
				if (idata[i])
					odata[num++] = idata[i];
			}
			timer().endCpuTimer();
			return num;
		}

		/**
		* CPU stream compaction using scan and scatter, like the parallel version.
		*
		* @returns the number of elements remaining after compaction.
		*/
		int compactWithScan(int n, int *odata, const int *idata) {
			if (n <= 0) return -1;
			int num = 0;
			// TODO
			//timer().startCpuTimer();
			for (int i = 0; i < n; i++) {
				odata[i] = idata[i] ? 1 : 0;
			}
			scan(n, odata, odata);
			num = odata[n - 1];
			for (int i = 0; i < n; i++) {
				if (idata[i])
					odata[odata[i]] = idata[i];
			}
			//timer().endCpuTimer();
			return num;
		}
	}
}
