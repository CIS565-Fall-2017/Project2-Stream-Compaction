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
		void scan(int n, int *odata, const int *idata, bool time) {

			if (time) timer().startCpuTimer();
			odata[0] = 0;

			//inclusive
			for (int i = 1; i < n; i++) {
				odata[i] = idata[i-1] + odata[i - 1];
			}

			if (time) timer().endCpuTimer();
		}
		
		void scan(int n, int *odata, const int *idata) {
			scan(n, odata, idata, true);
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
			int output = 0;

			timer().startCpuTimer();
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) odata[output++] = idata[i];
			}
			timer().endCpuTimer();

			return output;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			int *binaryValues = new int[n];
			int *scanValues = new int[n];
			int output = 0;

	        timer().startCpuTimer();

			//create binary array
			for (int i = 0; i < n; i++) {
				binaryValues[i] = (idata[i] == 0) ? 0 : 1;
			}

			//exclusive scan
			scan(n, scanValues, binaryValues, false);

			//populate odata
			for (int i = 0; i < n; i++) {
				if (binaryValues[i] == 1)
					odata[scanValues[i]] = idata[i];
			}

	        timer().endCpuTimer();
            
			output = binaryValues[n-1] == 0 ? scanValues[n-1] : scanValues[n-1] + 1;
			delete[] scanValues;
			delete[] binaryValues;
			return output;
        }
    }
}
