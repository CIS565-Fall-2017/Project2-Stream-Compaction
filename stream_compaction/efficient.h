#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

		void scanDynamicShared(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

		int compactDynamicShared(int n, int *odata, const int *idata);
    }
}
