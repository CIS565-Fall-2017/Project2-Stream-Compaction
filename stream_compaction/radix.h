#pragma once

#include "stream_compaction/common.h"


namespace StreamCompaction {
	namespace Radix {
		StreamCompaction::Common::PerformanceTimer& timer();


		void sort(int n, int * odata, int * idata);

	}
}