#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "sharedandbank.h"
namespace StreamCompaction {
	namespace SharedAndBank {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}

		void scan(const int n, int *odata, const int *idata) {


		}

		int compact(const int n, int *odata, const int *idata) {
			return -1;
		}
	}
}