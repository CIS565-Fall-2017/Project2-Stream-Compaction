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

		void scanImpl(int n, int *odata, const int *idata) {

			int pre;

			for (int i = 0; i < n; ++i)
			{

				if (i == 0) {
					pre = idata[i];
					odata[i] = 0;
				}
				else {
					int temp = idata[i];
					odata[i] = odata[i - 1] + pre;
					pre = temp;
				}
			}
		}

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			scanImpl(n, odata, idata);
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int count = 0;

			int k = 0;
			for (int i = 0; i < n; ++i)
			{
				if (idata[i] != 0)
				{
					count++;
					odata[k++] = idata[i];
				}
			}
	        timer().endCpuTimer();
            return -1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			for (int i = 0; i < n; ++i)
			{
				odata[i] = (idata[i] != 0);
			}

			scanImpl(n, odata, odata);

			int count = 0;
			for (int i = 0; i < n; ++i)
			{
				if (idata[i] != 0) {
					odata[odata[i]] = idata[i];
					count++;
				}
			}
	        timer().endCpuTimer();
            return -1;
        }
    }
}
