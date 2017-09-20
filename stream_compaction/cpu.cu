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
			if (n == 0)
			{
				return;
			}
	        timer().startCpuTimer();
            // TODO
			odata[0] = 0;
			for (int i = 1; i < n; i++)
			{
				odata[i] = idata[i-1] + odata[i-1];
			}

	        timer().endCpuTimer();
        }

		void scannotimer(int n, int *odata, const int *idata) {
			if (n == 0)
			{
				return;
			}
			odata[0] = 0;
			for (int i = 1; i < n; i++)
			{
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
            // TODO
			int num = 0;
			for (int i = 0; i < n; i++)
			{
				if (idata[i])
				{
					odata[num] = idata[i];
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
			//map
			int *mapped = (int*)malloc(n * sizeof(int));
			for (int i = 0; i < n; i++)
			{
				if (idata[i])
				{
					mapped[i] = 1;
				}
				else
				{
					mapped[i] = 0;
				}
			}
			//scan
			int *scanned = (int*)malloc(n * sizeof(int));
			scannotimer(n, scanned, mapped);
			//scatter
			int num = 0;
			for (int i = 0; i < n; i++)
			{
				if (mapped[i])
				{
					odata[scanned[i]] = idata[i];
					num++;
				}
			}
	        timer().endCpuTimer();
			return num;
        }
    }
}
