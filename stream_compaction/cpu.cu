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
		void scan_implementation(int n, int *odata, const int *idata) {
			if (n == 0)
				return;

			// The idea here is to be able to call scan in-place
			int prev = idata[0];
			odata[0] = 0;
			for (int i = 1; i < n; ++i)
			{
				int tmp = idata[i];
				odata[i] = prev + odata[i - 1];
				prev = tmp;
			}
		}

		void scan(int n, int *odata, const int *idata) {
			timer().startCpuTimer();
			scan_implementation(n, odata, idata);
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
		int compactWithoutScan(int n, int *odata, const int *idata) {
			timer().startCpuTimer();

			int sum = 0;
			for (int i = 0; i < n; ++i)
			{
				int tmp = idata[i] != 0 ? 1 : 0;
				odata[sum] = tmp * idata[i];
				sum += tmp;
			}
            
	        timer().endCpuTimer();
            return sum;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
		int compactWithScan(int n, int *odata, const int *idata) {
			timer().startCpuTimer();

			for (int i = 0; i < n; ++i)
				odata[i] = (idata[i] != 0 ? 1 : 0);

			// No malloc
			scan_implementation(n, odata, odata);

			// Scatter
			int sum = odata[n - 1];
			for (int i = 0; i < n; ++i)
				if (idata[i] != 0)
					odata[odata[i]] = idata[i];

	        timer().endCpuTimer();
            return sum;
        }
    }
}
