#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		void scan0(int n, int *odata, const int *idata);

        void scan(int n, int *odata, const int *idata);

		void scan_s(int n, int *odata, int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
