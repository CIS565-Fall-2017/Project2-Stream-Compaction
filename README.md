CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Mauricio Mutai
* Tested on: Windows 10, i7-7700HQ @ 2.2280GHz 16GB, GTX 1050Ti 4GB (Personal Computer)

### Overview

#### Introduction

The main aim of this project was to implement a few simple, but crucial GPU algorithms, understand their performance and how to improve it. The algorithms implemented were: exclusive scan, stream compaction, and radix sort.

For the exclusive scan (hereafter referred to as simply "scan"), multiple different implementations were made. A serial CPU version was created to check for correctness and as a comparison point. Then, a "naive" GPU version and a more "work-efficient" GPU version were implemented. Finally, a version of the work-efficient implementation was modified to use shared memory.

Below is a full list of the features implemented in this project:

* Serial CPU scan
* Naive GPU scan
* Work-efficient GPU scan with global memory (handles arbitrary arrays large enough to fit in the GPU's global memory)
* Work-efficient GPU scan with shared memory (only handles array as large as those that can fit in a single block's worth of threads)
* Thrust GPU scan
* GPU stream compaction
* GPU radix sort

### Performance Analysis

#### Rough optimization of block size

Below are two graphs showing some measurements of the runtime of the naive and work-efficient GPU scans, taken at different block sizes (that is, how many threads at most would be allocated to each block). The measurements were made with an array of size 1024.

![](img/block-naive.png)

![](img/block-workeff.png)

For the naive scan, the optimal block size appears to be 32, while for the work-efficient version, it is 16.

#### Comparison of GPU scans vs. CPU scans

Below is a graph showing the runtime of the CPU scan along with the naive, work-efficient with global memory, and Thrust GPU scans. The logarithmic x-axis shows the size of the array used for that measurement.

![](img/array-all-full.png)

Looking at just this view, it is hard to analyze the graph. Below is a view of one half of the graph, where the x-axis only goes up to 10000:

![](img/array-all-lower.png)



![](img/array-all-higher.png)
