/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/efficient_shared.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"
const int SIZE = 1 << 20; // must be <= 20
const int NPOT = SIZE - 3; // Non-Power-Of-Two
//int a[SIZE], b[SIZE], c[SIZE], d[SIZE];

int main(int argc, char* argv[]) {
    // Scan tests

	 int* a = reinterpret_cast<int*>(malloc(SIZE * sizeof(int)));
	 int* b = reinterpret_cast<int*>(malloc(SIZE * sizeof(int)));
	 int* c = reinterpret_cast<int*>(malloc(SIZE * sizeof(int)));
	 int* d = reinterpret_cast<int*>(malloc(SIZE * sizeof(int)));

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    //genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case

	for (int i = 0; i < SIZE; ++i) a[i] = i;
    //a[SIZE - 1] = 0;

    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("shared scan, power-of-two");
	StreamCompaction::Efficient_Shared::scan(SIZE, c, a);
	printElapsedTime(StreamCompaction::Efficient_Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(SIZE, c, true);
	printCmpResult(SIZE, b, c);

	zeroArray(SIZE, c);
	printDesc("shared scan, non-power-of-two");
	StreamCompaction::Efficient_Shared::scan(NPOT, c, a);
	printElapsedTime(StreamCompaction::Efficient_Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(NPOT, c, true);
	printCmpResult(NPOT, b, c);

	zeroArray(SIZE, c);
	//printDesc("thrust scan, power-of-two");
	StreamCompaction::Thrust::scan(SIZE, c, a);
	//printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	//printArray(SIZE, c, true);
	//printCmpResult(SIZE, b, c);
	
	zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int countSIZE, countNPOT, expectedSIZE, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    countSIZE = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
	expectedSIZE = countSIZE;
    printArray(expectedSIZE, b, true);
    printCmpLenResult(countSIZE, expectedSIZE, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
	countNPOT = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = countNPOT;
    printArray(countNPOT, c, true);
    printCmpLenResult(countNPOT, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
	countSIZE = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(countSIZE, c, true);
    printCmpLenResult(countSIZE, expectedSIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
	countSIZE = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(expectedSIZE, c, true);
    printCmpLenResult(countSIZE, expectedSIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
	countNPOT = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(expectedNPOT, c, true);
    printCmpLenResult(countNPOT, expectedNPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("shared compact, power-of-two");
	countSIZE = StreamCompaction::Efficient_Shared::compact(SIZE, c, a);
	printElapsedTime(StreamCompaction::Efficient_Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(expectedSIZE, c, true);
	printCmpLenResult(countSIZE, expectedSIZE, b, c);

	zeroArray(SIZE, c);
	printDesc("shared compact, non-power-of-two");
	countNPOT = StreamCompaction::Efficient_Shared::compact(NPOT, c, a);
	printElapsedTime(StreamCompaction::Efficient_Shared::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(expectedNPOT, c, true);
	printCmpLenResult(countNPOT, expectedNPOT, b, c);

 //   printf("\n");
 //   printf("*****************************\n");
 //   printf("** RADIX SORT TESTS **\n");
 //   printf("*****************************\n");

 //   // Radix Tests

	//int k = 4;
	//genArray(SIZE - 1, a, 1 << k); 
 //   printArray(SIZE, a, true);

 //   zeroArray(SIZE, b);
 //   printDesc("cpu sort, power-of-two");
 //   StreamCompaction::CPU::sort(SIZE, b, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   printArray(SIZE, b, true);

 //   zeroArray(SIZE, c);
 //   printDesc("cpu sort, non-power-of-two");
 //   StreamCompaction::CPU::sort(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   printArray(NPOT, c, true);

 //   zeroArray(SIZE, d);
 //   printDesc("radix sort, power-of-two");
 //   StreamCompaction::Radix::sort(SIZE, k + 1, d, a);
 //   printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   printArray(SIZE, d, true);
	//printCmpResult(SIZE, b, d);

 //   zeroArray(SIZE, d);
 //   printDesc("radix sort, non-power-of-two");
 //   StreamCompaction::Radix::sort(NPOT, k + 1, d, a);
 //   printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   printArray(NPOT, d, true);
 //   printCmpResult(NPOT, c, d);

    system("pause"); // stop Win32 console from closing on exit
}
