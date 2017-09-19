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
			//Actual implementation
			odata[0] = 0;
			for (int i = 1; i < n; ++i)
			{
				odata[i] = odata[i - 1] + idata[i - 1];
			}
		}

		void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
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
            // TODO		
			int num = 0;
			for (int i = 0; i < n; ++i)
			{
				if (idata[i] != 0)
				{
					odata[num] = idata[i];
					++num;
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
			
			int index;
			int *map = new int[n];
			int *scaned = new int[n];
			for (int i = 0; i < n; ++i){
				map[i] = (idata[i] == 0) ? 0 : 1;
			}

			scan_implementation(n, scaned, map);
			
			for (int i = 0; i < n; ++i) {
				if (map[i] == 1) {
					index = scaned[i];
					odata[index] = idata[i];
				}
			}
			
	        timer().endCpuTimer();
            return index + 1;
        }
    }
}
