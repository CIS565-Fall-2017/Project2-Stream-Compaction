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
		void scan_implementation(const int n, int* odata, const int* idata) {
			if (n <= 0) return;
			odata[0] = 0; //set first element to identity in exclusive scan
			for (int i = 1; i < n; ++i) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
		}
        void scan(const int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			scan_implementation(n, odata, idata);
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(const int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int index = 0;
			for (int i = 0; i < n; ++i) {
				if (idata[i] != 0) { odata[index++] = idata[i]; }
			}
	        timer().endCpuTimer();
			return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(const int n, int *odata, const int *idata) {

			int* bools = new int[n];
			int* indices = new int[n];

	        timer().startCpuTimer();
			for (int i = 0; i < n; ++i) {
				bools[i] = idata[i] == 0 ? 0 : 1;
			}
			scan_implementation(n, indices, bools);
			int size = bools[n - 1] + indices[n - 1];

			for (int i = 0; i < n; ++i) {
				if (bools[i] == 1) { odata[indices[i]] = idata[i]; }
			}
	        timer().endCpuTimer();

			delete[] bools;
			delete[] indices;
			return size;
        }
    }
}
