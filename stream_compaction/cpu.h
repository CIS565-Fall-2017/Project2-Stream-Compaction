#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
	    StreamCompaction::Common::PerformanceTimer& timer();

		void scanImplementation(int n, int *odata, const int *idata);

        void scan(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);

		void quickSort(int n, int *odata, const int *idata);
    }
}
