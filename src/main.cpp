/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <thread>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 22; // feel free to change the size of array
const int NPOT = SIZE - 5; // Non-Power-Of-Two
int a[SIZE], b[SIZE], c[SIZE];

int testInput[] = { 1, 5, 0, 1, 2, 0, 3 };
int testOutput[] = { 0, 1, 6, 6, 7, 9, 9 };

int testCompactionInput[] = { 1, 5, 0, 1, 2, 0, 3 };
int testCompactionOutput[] = { 1, 5, 1, 2, 3 };

int main(int argc, char* argv[]) {

	printf("SIZE: %d", SIZE);

    // Scan tests
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

	// Test example
	zeroArray(7, b);
	printDesc("cpu scan, results test");
	StreamCompaction::CPU::scan(7, b, testInput);
	bool pass = true;
	for (int i = 0; i < 7; ++i)
		if (testOutput[i] != b[i])
			pass = false;
	printDesc((std::string("PASS: ") + (pass ? "YES": "NO")).c_str());
	printArray(7, testOutput, true);
	printArray(7, b, true);

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

	// Test results
	zeroArray(SIZE, b);
	printDesc("cpu compact without scan, power-of-two");
	count = StreamCompaction::CPU::compactWithoutScan(7, b, testCompactionInput);
	pass = cmpArrays(5, testCompactionOutput, b) == 0;
	printDesc((std::string("PASS: ") + (pass ? "YES" : "NO")).c_str());
	expectedCount = count;
	printArray(count, b, true);
	printArray(5, testCompactionOutput, true);

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

	bool generateCSV = false;

	if (generateCSV)
	{
		bool useNPOT = true;
		int steps = 22;		
		std::vector<std::vector<float>> timeData;

		for (int i = 1; i < steps + 1; ++i)
		{
			int size = (1 << i);

			if (useNPOT)
				size = (size - 3 > 0) ? size - 3 : size;

			int * data = new int[size];
			int * result = new int[size];
			genArray(size, data, i * 5);
			zeroArray(size, data);

			std::vector<float> stepData;
			stepData.push_back(size);

			// Run each implementation -- we don't care about the results (the previous tests cover that)
			{
				zeroArray(size, result);
				StreamCompaction::CPU::scan(size, result, data);
				stepData.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());

				zeroArray(size, result);
				StreamCompaction::Naive::scan(size, result, data);
				stepData.push_back(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());

				zeroArray(size, result);
				StreamCompaction::Efficient::scan(size, result, data);
				stepData.push_back(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());

				zeroArray(size, result);
				StreamCompaction::Thrust::scan(size, result, data);
				stepData.push_back(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
			}

			timeData.push_back(stepData);

			delete[] data;
			delete[] result;
		}

		std::ofstream fstr;
		fstr.open("data.csv", std::ofstream::out);

		for (int i = 0; i < timeData.size(); ++i)
		{
			std::string line = "";

			for (int j = 0; j < timeData[i].size(); ++j)
				line += std::to_string(timeData[i][j]) + ", "; // Parsers remove this

			line += "\n";
			std::cout << line << std::endl;
			fstr.write(line.c_str(), line.length());
		}

		fstr.close();
	}


    system("pause"); // stop Win32 console from closing on exit
}
