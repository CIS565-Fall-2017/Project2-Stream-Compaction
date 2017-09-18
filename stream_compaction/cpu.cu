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
			//TODO
			scanImplementation(n, odata, idata);
			timer().endCpuTimer();

        }

		void scanImplementation(int n, int *odata, const int *idata) {
			odata[0] = 0;
			for (int i = 1; i < n; ++i) {
				odata[i] = idata[i - 1] + odata[i - 1];
			}
		}

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[count] = idata[i];
					count++;
				}
			}
	        timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			int *temp = new int[n];
			int *temp2 = new int[n];
			timer().startCpuTimer();
	        // TODO
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					temp[i] = 1;
				}
				else {
					temp[i] = 0;
				}
				temp2[i] = 0;
			}
			// TODO: Figure out how to not call timer twice XD
			scanImplementation(n, temp2, temp);
			for (int i = 0; i <= n; i++) {
				if (temp[i] == 1) {
					odata[temp2[i]] = idata[i];
				}
			}
			int count = temp2[n - 1];
			delete[] temp;
			delete[] temp2;
	        timer().endCpuTimer();
			return count;
        }
    }
}
