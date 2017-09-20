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
#include "testing_helpers.hpp"

const int SIZE = 1 << 15; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int a[SIZE], b[SIZE], c[SIZE];

const int const sizes[10] = { 7,8,9,10,11,12,13,14,15,16 };

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
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
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

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
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);


	printf("\n");
	printf("*****************************\n");
	printf("** MY OWN RADIX SORT TESTS **\n");
	printf("*****************************\n");
	printDesc("radix basic, power-of-two");
	const int COUNT_BASIC = 16;
	int basic[COUNT_BASIC];
	int basic_out[COUNT_BASIC];
	genArray(COUNT_BASIC, basic, 78);
	StreamCompaction::Radix::sort(COUNT_BASIC, basic_out, basic);
	//printCPUArray(COUNT_BASIC, basic_out);
	printf("\n");
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)\n\n");

	printDesc("radix basic, non power-of-two");
	const int COUNT_BASIC2 = 18;
	int basic2[COUNT_BASIC2];
	int basic_out2[COUNT_BASIC2];
	genArray(COUNT_BASIC2, basic2, 78);
	StreamCompaction::Radix::sort(COUNT_BASIC2, basic_out2, basic2);
	//printCPUArray(COUNT_BASIC2, basic_out2);
	printf("\n");
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)\n");


	printDesc("radix massive, power-of-two");
	const int COUNT_MASSIVE = 1 << 10;
	int massive[COUNT_MASSIVE];
	int massive_out[COUNT_MASSIVE];
	genArray(COUNT_MASSIVE, massive, 78);
	StreamCompaction::Radix::sort(COUNT_MASSIVE, massive_out, massive);
	//printCPUArray(COUNT_MASSIVE, massive_out);
	printf("\n");
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)\n\n");

	printDesc("radix massive, non power-of-two");
	const int COUNT_MASSIVE2 = (1 << 10) - 6;
	int massive2[COUNT_MASSIVE2];
	int massive_out2[COUNT_MASSIVE2];
	genArray(COUNT_MASSIVE2, massive2, 78);
	StreamCompaction::Radix::sort(COUNT_MASSIVE2, massive_out2, massive2);
	//printCPUArray(COUNT_MASSIVE2, massive_out2);
	printf("\n");
	printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)\n");



	for (int i = 7; i < 16; i++) {
		printf("\n*********** WITH %d SIZED ARRAY *********\n", 1 << i);
		const int COUNT_MASSIVE = 1 << i;
		printDesc("radix massive, power-of-two");

		int* massive = (int*) malloc(sizeof(int) * COUNT_MASSIVE);
		int* massive_out = (int*) malloc(sizeof(int) * COUNT_MASSIVE);
		genArray(COUNT_MASSIVE, massive, 78);
		StreamCompaction::Radix::sort(COUNT_MASSIVE, massive_out, massive);
		//printCPUArray(COUNT_MASSIVE, massive_out);
		printf("\n");
		printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)\n\n");

		printDesc("radix massive, non power-of-two");
		const int COUNT_MASSIVE2 = (1 << i) - 6;
		int* massive2 = (int*)malloc(sizeof(int) * COUNT_MASSIVE2);
		int* massive_out2 = (int*)malloc(sizeof(int) * COUNT_MASSIVE2);
		genArray(COUNT_MASSIVE2, massive2, 78);
		StreamCompaction::Radix::sort(COUNT_MASSIVE2, massive_out2, massive2);
		//printCPUArray(COUNT_MASSIVE2, massive_out2);
		printf("\n");
		printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)\n");

	}

    system("pause"); // stop Win32 console from closing on exit
}
