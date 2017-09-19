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

		void scanImplementation(int n, int *odata, const int *idata) {
			// an EXCLUSIVE scan
			odata[0] = 0;
			for (int k = 1; k < n; k++) {
				odata[k] = odata[k - 1] + idata[k - 1];
			}
		}


        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO	
			scanImplementation(n, odata, idata);

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
			int count = 0;

			timer().startCpuTimer();
            // TODO

			for (int k = 0; k < n; k++) {
				if (idata[k] != 0) {
					odata[count++] = idata[k];
				}
			}

	        timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			
			int* tempArray = new int[n];
			int* scanResult = new int[n];
			int count = 0;

			timer().startCpuTimer();
	        // TODO
			
			// Step 1 : compute temp array
			for (int k = 0; k < n; k++) {
				tempArray[k] = idata[k] ? 1 : 0;
			}

			// Step 2 : run exclusive on temp array
			scanImplementation(n, scanResult, tempArray);

			// Step 3 : scatter
			count = tempArray[n - 1] ? (scanResult[n - 1] + 1) : scanResult[n - 1];

			for (int k = 0; k < n; k++) {
				if (tempArray[k]) {
					odata[scanResult[k]] = idata[k];
				}
			}

	        timer().endCpuTimer();

			delete[] tempArray;
			delete[] scanResult;

            return count;

        }

		int compare(const void * a, const void * b)
		{
			return (*(int*)a - *(int*)b);
		}

		void quickSort(int n, int *odata, const int *idata){

			for (int k = 0; k < n; k++) {
				odata[k] = idata[k];
			}

			timer().startCpuTimer();
			
			qsort(odata, n, sizeof(int), compare);

			timer().endCpuTimer();
		}
    }
}
