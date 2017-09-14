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

            // TODO
			int sum = 0;
			for (int i = 0; i < n; ++i ) {
				odata[i] = sum;
				sum += idata[i];
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

            // TODO
			int num = 0;
			for (int i = 0; i < n; ++i) {
				int val = idata[i];
				if (val != 0) {
					odata[num] = val;
					num++;
				}
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
	        timer().startCpuTimer();
	        // TODO

			int* temp = new int[n + 1];
			temp[n] = 0;
			int num = 0;
			// compute temporary array
			for (int i = 0; i < n; ++i) {
				temp[i] = (idata[i] == 0) ? 0 : 1;
			}
			// run exclusive scan on temporary array
			int sum = 0;
			for (int i = 0; i <= n; ++i) {
				int val = temp[i];
				temp[i] = sum;
				sum += val;
			}

			// scatter
			for (int i = 1; i <= n; ++i) {
				if (temp[i] != temp[i-1]) {
					odata[num] = idata[i - 1];
					num++;
				}
			}

			delete[] temp;
	        timer().endCpuTimer();
            return num;
        }
    }
}
