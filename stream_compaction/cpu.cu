#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        bool compactTest = false;
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
          if (!compactTest) timer().startCpuTimer();
          odata[0] = 0;
          for (size_t k = 1; k < n; ++k)
            odata[k] = odata[k - 1] + idata[k-1];
          if (!compactTest) timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          int odataIdx = 0;
          for (size_t k = 0; k < n; ++k)
            if (idata[k] != 0)
              odata[odataIdx++] = idata[k];
	        timer().endCpuTimer();
          return odataIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
          compactTest = true;
          int* tdata = new int[n];
          int* sdata = new int[n];

	        timer().startCpuTimer();

          for (size_t k = 0; k < n; ++k)
            if (idata[k] != 0)
              tdata[k] = 1;
            else
              tdata[k] = 0;

          scan(n, sdata, tdata);

          int sdataLastIdx = 0;
          for (size_t k = 0; k < n; ++k) {
            if (tdata[k] == 1) {
              odata[sdata[k]] = idata[k];
              sdataLastIdx = sdata[k];
            }
          }

	        timer().endCpuTimer();

          compactTest = false;
          delete[] tdata, sdata;
          return sdataLastIdx+1;
        }
    }
}
