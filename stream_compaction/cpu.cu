#include <cstdio>
#include "cpu.h"

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
	        timer().startCpuTimer();
			if (n <= 0) return;
			odata[0] = 0; //set first element to identity in exclusive scan
			for (int i = 1; i < n; ++i) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int index = 0;
			for (int i = 0; i < n; ++i) {
				if (idata[i] != 0) odata[index++] = idata[i];
			}
	        timer().endCpuTimer();
			return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			int* isNonZero = new int[n];
			int* isNonZeroScan = new int[n];
			for (int i = 0; i < n; ++i) {
				isNonZero[i] = idata[i] == 0 ? 0 : 1;
			}
			scan(n, isNonZeroScan, isNonZero);
			int size = isNonZero[n - 1] + isNonZeroScan[n - 1];

			int index = 0;
			for (int i = 0; i < n; ++i) {
				if (isNonZero[i] == 1) odata[isNonZeroScan[i]] = idata[i];
			}
			delete[] isNonZero;
			delete[] isNonZeroScan;

	        timer().endCpuTimer();
			return size;
        }
    }
}
