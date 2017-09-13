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
			//TODO
			int sum = 0;
			for (int i = 0; i < n; i++)
			{
				odata[i] = sum;
				sum += idata[i];	
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
			int numel = 0;
			for (int i = 0; i < n; i++)
			{
				if (idata[i] != 0)
				{
					odata[numel] = idata[i];
					numel++;
				}
			}
	        timer().endCpuTimer();
            return numel;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			int *oscan = (int*)malloc(n * sizeof(int));
			int *iscan = (int*)malloc(n * sizeof(int));
			for (int i = 0; i < n; i++)
				if(idata[i]==0)
					iscan[i] = 0;
				else
					iscan[i] = 1;
			scan(n, oscan, iscan);
			int numel = 0;
			for (int i = 0; i < n; i++)
			{
				if (idata[i] != 0)
				{
					odata[oscan[i]] = idata[i];
					numel++;
				}
			}
	        timer().endCpuTimer();
            return numel;
        }
    }
}
