#pragma once

#include "common.h"

namespace RadixSort {
	StreamCompaction::Common::PerformanceTimer& timer();

	void sort(int n, int numOfBits, int *odata, const int *idata);
}