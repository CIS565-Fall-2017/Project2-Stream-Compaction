
#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedAndBank {
        StreamCompaction::Common::PerformanceTimer& timer();

		__global__ void kernZeroExcessLeaves(const int pow2roundedsize, const int orig_size, int* data);
		__global__ void kernScan(const int shMemEntries, int* odata, const int* idata, int* SUMS);
		__global__ void kernAddBack(const int n, int* odata, const int* scannedSumsLevel);
		void recursiveScan(const int n, const int level, int *odata, const int *idata);
        void scan(const int n, int *odata, const int *idata);
        int compact(const int n, int *odata, const int *idata);
    }
}
