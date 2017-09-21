#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedAndIndexing {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
