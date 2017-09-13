CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ricky Rajani
* Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, NVIDIA Quadro K620 (Moore 100C Lab)

This project implements GPU stream compaction in CUDA, from scratch. The algorithm will later be used for acceleration a path trace project. The algorithms implemented take advantage of the GPU's massive parrallelism, specfically data parrallelism.

Different versions of the Scan (Prefix Sum) algorithm were implemented:
- CPU version
- "Naive"
- "Work-efficient"
- GPU stream compaction

### Performance Analysis

Optimal blocksize = 128
Blocksizes from 32 to 1024 (exponentially) were tried on the implementations. The optimal blocksize value was 128 and 256.

![](img/GraphComparisons.PNG)

There wasn't a significant improvement in performance from Naive to Work-Efficient implementations until an array size of 2^16. This could be due to fewer calls to kernels in the Naive implementation. However, the Work-Efficient implementation is faster as the array size increases due to less computations.

Scan Test Results:

![](img/ScanTests.PNG)

Stream Compaction Test Results:

![](img/CompactionTests.PNG)
