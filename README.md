CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Name: Meghana Seshadri
* Tested on: Windows 10, i7-4870HQ @ 2.50GHz 16GB, GeForce GT 750M 2048MB (personal computer)


## Project Overview

The goal of this project was to get an introduction to the GPU stream compaction in CUDA. This algorithm is often used to quickly process large sets of data according to certain conditions - hence very useful in parallel programming. In this project, the stream compaction will remove zero's from an array of int's. 


The following four versions of stream compaction were implemented:

* CPU algorithm
* Naive parallel algorithm 
* Work-efficient algorithm 
* Thrust-based algorithm


The stream compaction algorithm breaks down into a scan function and a compat function. The scan function computes an exclusive prefix sum. The compact function maps an input array to an array of zero's and one's, scans it, and then uses scatter to produce the output. 

The following descriptions dive a bit deeper into the different approaches:

### CPU
* `n - 1` adds for an array of length `n`

### Naive Parallel


### Work-Efficient


### Thrust-Based


## Performance Analysis 

### Block Comparison

Below is a graph that shows which would be the best block size to test on. The times below are for the work-efficient algorithm, comparing 64 elements and 65536 elements.

![](images/blockcomparison.PNG)

Based on the chart above, block size 128 on average, seems to be the most efficient. The rest of the analysis below will be using that block size.


### Algorithm Comparison

Below are charts showing tests across all 4 algorithms, both testing with power-of-two sized arrays and non-power-of-two sized arrays.


![](images/algorithmcomparison.PNG)


![](images/algorithmcomparisontable.PNG)



### Scan and Stream Compaction Test Results

The following tests were run on 128 elements at a block size of 128.

```
****************
** SCAN TESTS **
****************
    [  40  12   3  21  27   2  32  20   1   4  41  34  15 ...  34   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.000821ms    (std::chrono Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6112 6146 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.000821ms    (std::chrono Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6063 6064 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.44064ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6112 6146 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.433024ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6063 6064 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.781184ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6112 6146 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 1.20867ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6099 6099 ]
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6063 6064 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.34442ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6112 6146 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.019392ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6063 6064 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   0   1   3   1   0   0   2   1   0   1   0   1 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.000821ms    (std::chrono Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.000821ms    (std::chrono Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.962322ms    (std::chrono Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 2.22262ms    (CUDA Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   1   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 2.10656ms    (CUDA Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   3   1 ]
    passed
Press any key to continue . . .
```