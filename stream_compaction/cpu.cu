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
			
			// TODO
			

			int* Temp = new int[n];
			memcpy(Temp, idata, sizeof(int)* n);

			timer().startCpuTimer();

			for (int k = 1; k < n; k++)
			{				
				Temp[k] += Temp[k-1];
			}

			//Inclusive to Exclusive
			memcpy(&odata[1], Temp, sizeof(int)* (n - 1));
			odata[0] = 0; //Identity

			timer().endCpuTimer();

			delete [] Temp;

			
			
		}

		/**
		* CPU stream compaction without using the scan function.
		*
		* @returns the number of elements remaining after compaction.
		*/
		int compactWithoutScan(int n, int *odata, const int *idata) {

			timer().startCpuTimer();
			// TODO			

			int counter = 0;
			for (int i = 0; i < n; i++)
			{
				if (idata[i] != 0)
				{
					odata[counter++] = idata[i];
				}
			}

			timer().endCpuTimer();
			return counter;
		}

		/**
		* CPU stream compaction using scan and scatter, like the parallel version.
		*
		* @returns the number of elements remaining after compaction.
		*/
		int compactWithScan(int n, int *odata, const int *idata) {
			
			
			
			// TODO
			int* Map = new int[n];
			int* tempScan = new int[n];
			int* ArrayforScan = new int[n];

			timer().startCpuTimer();

			//Mapping
			for (int i = 0; i < n; i++)
			{
				if (idata[i] != 0)
					Map[i] = 1;
				else
					Map[i] = 0;
			}

			memcpy(tempScan, Map, sizeof(int)* n);

			//Scan
			for (int k = 1; k < n; k++)
			{
				tempScan[k] += tempScan[k - 1];
			}

			//Inclusive to Exclusive
			memcpy(&ArrayforScan[1], tempScan, sizeof(int)* (n - 1));
			ArrayforScan[0] = 0; //Identity

			//Scatter
			int counter = 0;
			for (int i = 0; i < n; i++)
			{
				if (Map[i] != 0)
				{
					odata[ArrayforScan[i]] = idata[i];
					counter++;
				}
			}

			timer().endCpuTimer();

			delete[] Map;
			delete[] tempScan;
			delete[] ArrayforScan;

			

			return counter;
		}
	}
}
