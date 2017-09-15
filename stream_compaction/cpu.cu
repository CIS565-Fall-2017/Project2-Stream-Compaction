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

          odata[0] = 0;
          for (int i = 1; i < n; ++i)
          {
            odata[i] = idata[i - 1] + odata[i - 1];
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
          int index = 0;
          for (int i = 0; i < n; ++i)
          {
            if (idata[i])
            {
              odata[index] = idata[i];
              index++;
            }
          }
          timer().endCpuTimer();
          return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

          int* tempScanArray = (int*) malloc(sizeof(int) * n);
          int* tempScanResultArray = (int*) malloc(sizeof(int) * n);

          // Create temporary array
          for (int i = 0; i < n; ++i)
          {
            tempScanArray[i] = (idata[i]) ? 1 : 0;
          }

          // Included exclusive scan implementation here in order to avoid the conflict with multiple timers:
          tempScanResultArray[0] = 0;
          for (int i = 1; i < n; ++i)
          {
            tempScanResultArray[i] = tempScanArray[i - 1] + tempScanResultArray[i - 1];
          }

          // Scatter
          for (int i = 0; i < n; ++i)
          {
            if (tempScanArray[i])
            {
              odata[tempScanResultArray[i]] = idata[i];
            }
          }

          int compactSize = (tempScanArray[n - 1]) ? tempScanResultArray[n - 1] + 1 : tempScanResultArray[n - 1];
          timer().endCpuTimer();
          return compactSize;
        }
    }
}
