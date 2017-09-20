CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ricky Rajani
* Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, NVIDIA Quadro K620 (Moore 100C Lab)

This project implements GPU stream compaction in CUDA, from scratch. The algorithm will later be used for accelerating a path tracer project. The algorithms implemented take advantage of the GPU's massive parrallelism, specfically data parrallelism.

Different versions of the Scan (Prefix Sum) algorithm were implemented:
- CPU version
- Naive
- Work-efficient
- Thrust

### Performance Analysis

Optimal blocksize = 128

Blocksizes from 32 to 1024 (exponentially) were tried on the implementations. The optimal blocksize values were 128 and 256.

![](img/graph-comp.PNG)

There wasn't a significant improvement in performance from Naive to Work-Efficient implementations until an array size of 2^16. This could be due to fewer calls to kernels in the Naive implementation. However, the Work-Efficient implementation is faster as the array size increases due to less computations. Both of the GPU implementations start to plateau as the array size increases while the CPU implementation's time increases exponentially. However, with the array sizes I experimented with, the CPU implementation is always significantly faster than the Naive and Work-Efficient implementations. 

Array size: 2^8

Scan Test Results:

![](img/scan-tests.PNG)

Stream Compaction Test Results:

![](img/compaction-tests.PNG)
