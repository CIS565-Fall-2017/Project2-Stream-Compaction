
#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedAndBank {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(const int n, int *odata, const int *idata);
        int compact(const int n, int *odata, const int *idata);
    }
}
