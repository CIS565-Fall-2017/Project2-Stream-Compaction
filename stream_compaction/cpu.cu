#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
namespace CPU {

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
  static PerformanceTimer timer;
  return timer;
}

inline void scan_impl(int n, int *odata, const int *idata) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    *odata++ = sum;
    sum += *idata++;
  }
}

/**
 * CPU scan (prefix sum).
 * For performance analysis, this is supposed to be a simple for loop.
 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
 */
void scan(int n, int *odata, const int *idata) {
  timer().startCpuTimer();
  scan_impl(n, odata, idata);
  timer().endCpuTimer();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
  timer().startCpuTimer();
  int count = 0;
  for (int i = 0; i < n; ++i) {
    int d = *idata++;
    if (d) {
      *odata++ = d;
      ++count;
    }
  }
  timer().endCpuTimer();
  return count;
}

void scatter(int n, int *output, const int *map, const int *input) {
  for (int i = 0; i < n; ++i) {
    output[map[i]] = input[i];
  }
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
  timer().startCpuTimer();
  int *temp = (int*)malloc(sizeof(int)*n);
  int *temp2 = (int*)malloc(sizeof(int)*n);
  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (idata[i]) {
      temp[i] = 1;
      ++count;
    } else {
      temp[i] = 0;
    }
  }
  scan_impl(n, temp2, temp);
  scatter(n, odata, temp2, idata);
  timer().endCpuTimer();
  return count;
}

} // namespace CPU
} // namespace StreamCompaction
