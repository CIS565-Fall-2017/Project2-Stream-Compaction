#include <cstdio>
#include <iostream>
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
			odata[0] = 0;
			for (int i = 1;i < n;i++)
			{
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
            // TODO
			int countOut = 0;
			for (int tempCount = 0; tempCount <n; tempCount++)
			{
				if (idata[tempCount] != 0)
				{
					odata[countOut] = idata[tempCount];
					countOut++;
				}
			}
	        timer().endCpuTimer();
            return countOut;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			int count = 0;
			odata[0] = 0;
			for (int i = 1;i < n;i++)
			{
				odata[i] = odata[i - 1] + idata[i - 1];
			}

			for (int j = 0;j < n - 1;j++)
			{
				if (odata[j] != odata[j + 1])
				{
					odata[count] = idata[j];
					count++;
				}
			}
			
	        timer().endCpuTimer();
			return count;
        }
    }
}
