#pragma once

#include "common.h"
#include <vector>

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

		void radixScan(int n, int *odata, const int *idata);

		void sortArray(int n, int *b, int *a);
		
    }
}
