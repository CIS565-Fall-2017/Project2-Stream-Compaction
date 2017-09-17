#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "RadixSort.h"

namespace RadixSort {
	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	void sort(int n, int numOfBits, int *odata, const int *idata) {

	}
}