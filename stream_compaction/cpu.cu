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
	        //timer().startCpuTimer();

          odata[0] = 0;

          for (int i = 1; i < n; i++)
          {
            odata[i] = odata[i - 1] + idata[i - 1];
          }

	        //timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
          int j = 0;

	        timer().startCpuTimer();

          for (int i = 0; i < n; i++)
          {
            if (idata[i] != 0)
            {
              odata[j++] = idata[i];
            }
          }

	        timer().endCpuTimer();

          return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
          int *scanResult = new int[n];
          int j = 0;

	        timer().startCpuTimer();

          for (int i = 0; i < n; i++)
          {
            odata[i] = idata[i] == 0 ? 0 : 1;
          }

          scan(n, scanResult, odata);

          for (int i = 0; i < n-1; i++)
          {
            if (odata[i] == 1)
            {
              odata[scanResult[i]] = idata[i];
            }
          }

          if (odata[n - 1] == 1)
          {
            odata[scanResult[n - 1] + 1] = idata[n - 1];
          }

	        timer().endCpuTimer();

          free(scanResult);
          return j;
        }
    }
}
