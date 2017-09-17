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
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/sharedandbank.h>
#include "testing_helpers.hpp"

//SIZEs: 8, 16, 20, 24(max on this laptop)
const int SIZE = 1 << 24; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
const int range = 50;
int a[SIZE], b[SIZE], c[SIZE], sortRef[SIZE], sortRef_npot[NPOT], sort[SIZE];

int main(int argc, char* argv[]) {

    genArray(SIZE - 1, a, range);  
    a[SIZE - 1] = 0;// Leave a 0 at the end to test that edge case
    printArray(SIZE, a, true);

    printf("SIZE: %i\n", SIZE);
    printf("\n");
    printf("*****************************\n");
    printf("**** BEGIN RADIX TESTS ******\n");
    printf("*****************************\n");

    copyArray(SIZE, sortRef, a);
    printDesc("std::sort, power-of-two(full array)");
	StreamCompaction::CPU::timer().startCpuTimer();
	std::sort(sortRef, sortRef + SIZE);
	StreamCompaction::CPU::timer().endCpuTimer();
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, sortRef, true);

    copyArray(NPOT, sortRef_npot, a);
    printDesc("std::sort, non-power-of-two");
	StreamCompaction::CPU::timer().startCpuTimer();
	std::sort(sortRef_npot, sortRef_npot + NPOT);
	StreamCompaction::CPU::timer().endCpuTimer();
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, sortRef_npot, true);

    copyArray(SIZE, sort, a);
    printDesc("CPU radix sort, power-of-two");
	StreamCompaction::CPU::radixSort(SIZE, ilog2ceil(range), sort);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, sort, true);
    printCmpResult(SIZE, sortRef, sort);

    copyArray(NPOT, sort, a);
    printDesc("CPU radix sort, non power-of-two");
	StreamCompaction::CPU::radixSort(NPOT, ilog2ceil(range), sort);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, sort, true);
    printCmpResult(NPOT, sortRef_npot, sort);

    copyArray(SIZE, sort, a);
    printDesc("GPU radix sort, power-of-two");
	StreamCompaction::Radix::radixSort(SIZE, ilog2ceil(range), sort);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, sort, true);
    printCmpResult(SIZE, sortRef, sort);

    copyArray(NPOT, sort, a);
    printDesc("GPU radix sort, non power-of-two");
	StreamCompaction::Radix::radixSort(NPOT, ilog2ceil(range), sort);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, sort, true);
    printCmpResult(NPOT, sortRef_npot, sort);


    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

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
    printArray(SIZE, c, true);
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

    //////////zeroArray(SIZE, c);
    //////////printDesc("work-efficient shared mem and bank conflict free scan, power-of-two");
    //////////StreamCompaction::SharedAndBank::scan(SIZE, c, a);
    //////////printElapsedTime(StreamCompaction::SharedAndBank::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //////////printArray(SIZE, c, true);
    //////////printCmpResult(SIZE, b, c);

    //////////zeroArray(SIZE, c);
    //////////printDesc("work-efficient shared mem and bank conflict free scan, non-power-of-two");
    //////////StreamCompaction::SharedAndBank::scan(NPOT, c, a);
    //////////printElapsedTime(StreamCompaction::SharedAndBank::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //////////printArray(NPOT, c, true);
    //////////printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);
    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    //system("pause"); // stop Win32 console from closing on exit
}
