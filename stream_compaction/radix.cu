#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"

namespace StreamCompaction {
	namespace Radix {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}

		void sort(const int n, int *odata, const int *idata) {


		}

	}
}