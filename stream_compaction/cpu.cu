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

		void scanhelp(int n, int *odata, const int *idata) 
		{
			odata[0] = 0;
			for (int i = 1; i < n; i++) 
			{
				odata[i] = odata[i - 1] + idata[i - 1];
			}
		}



        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
			scanhelp(n, odata, idata);

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {

			int counter = 0;
	        timer().startCpuTimer();
            // TODO
			for (int i = 0; i < n; i++) 
			{
				if (idata[i] != 0) 
				{
					odata[counter++] = idata[i];
				}
			}
			
	        timer().endCpuTimer();
			return counter;
            //return -1;
        }



		int scatter(int n, int *odata, const int *idata, const int *ichange, const int *exSum) 
		{
			int counter = 0;
			for (int i = 0; i < n; i++) 
			{
				if (ichange[i] == 1) 
				{
					odata[exSum[i]] = idata[i];
					counter++;
				}
			}
			return counter;
		}



        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

			int* iChange = new int[n];
			int* exSum = new int[n];

	        timer().startCpuTimer();
	        // TODO

			for (int i = 0; i < n; i++) 
			{
				iChange[i] = (idata[i] == 0) ? 0 : 1;
			}

			//odataChanged is the exclusive prefix sum
			scanhelp(n, exSum, iChange);
			int counter = scatter(n, odata, idata, iChange, exSum);

			timer().endCpuTimer();

			delete[] iChange;
			delete[] exSum;


			return counter;
        }
    }
}
