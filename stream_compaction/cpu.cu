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
			int i;
			int count=0;

			for (i = 0; i < n; i++)
			{
				count += idata[i];
				odata[i] = count;
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
			int i;
			int count = 0;
			for (i = 0; i < n; i++)
			{
				if (idata[i] != 0)
				{
					odata[i-count] = idata[i];				
				}
				else
				{
					count++;
				}
			}
	        timer().endCpuTimer();
            return n-count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			int i;
			int count=0;

			for (i = 0; i < n; i++)
			{
				if (idata[i] != 0)
				{
					odata[i - count] = idata[i];
				}
				else
				{
					count++;
				}
			}

	        timer().endCpuTimer();
            return n- count;
        }
    }
}
