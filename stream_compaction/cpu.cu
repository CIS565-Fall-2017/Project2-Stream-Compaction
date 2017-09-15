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
			bool timerStarted = false;
			try {
				timer().startCpuTimer();
			}
			catch (const std::runtime_error& e) {
				timerStarted = true;
			}

			int sum = 0;
			for (int i = 0; i < n; i++) {
				odata[i] = sum;
				sum += idata[i];
			}

			if (!timerStarted) {
				timer().endCpuTimer();
			}
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int counter = 0;
			for (int i = 0; i < n; i++) {
				int input = idata[i];
				if (input != 0) {
					odata[counter] = input;
					counter++;
				}
			}
			odata[counter] = '\0';
	        timer().endCpuTimer();

            return counter;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			// Map to zeros and ones
			int *onesAndZeros = (int*)malloc(n * sizeof(int));
			for (int i = 0; i < n; i++) {
				onesAndZeros[i] = (idata[i] == 0) ? 0 : 1;
			}

			// Scan
			int *scanned = (int*)malloc(n * sizeof(int));
			scan(n, scanned, onesAndZeros);

			// Scatter
			int counter = 0;
			for (int i = 0; i < n; i++) {
				if (onesAndZeros[i] == 1) {
					int index = scanned[i];
					odata[index] = idata[i];
					counter++;
				}
			}
			odata[counter] = '\0';

			// Free memory
			free(scanned);
			free(onesAndZeros);

	        timer().endCpuTimer();

            return counter;
        }
    }
}
