#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(const int n, int *odata, const int *idata);
        void scan_notimer(const int n, int *odata, const int *idata);

        int compact(const int n, int *odata, const int *idata);
    }
}
