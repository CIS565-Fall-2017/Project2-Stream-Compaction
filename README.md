CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Name: William Ho
* Email: willho@seas.upenn.edu
* Tested on: Windows 7 Professional, i7-6700 @ 3.40 GHz 16.0GB, NVIDIA QuadroK620 (Moore 100C Lab)

## Overview

**Stream Compaction** is a process for culling elements from an array that do not meet a certain criteria. It is a widely used technique for compression, utilized for audio compression, and has many use cases for computer graphics, such as culling unnecessary rays from a path tracer. A conceptually simple example is show below, in which we remove `0`s from an array of `int`s:

* Starting Array: {0, 2, 3, 0, 5, 2, 0, 0, 9}

* Compacted Array: {2, 3, 5, 2, 9}

The goal for this project was to observe a few variations on the Stream Compaction algorithm, note that it is a parallelizable technique that can be implemented on the GPU, and use it as a case study to gain an understanding of working on the GPU. Within Stream Compaction, a common implementation technique involves "scanning" an array and creating a corresponding array of exclusive prefix-sums. 

I implement **Scan**, (and using it, Stream Compaction) with a straightforward CPU implementation, a naively parallelized CUDA implementation on the GPU, and a work-efficient CUDA implementation, the latter two of which are adapted from GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html).

## Implementation Details

### Scan

### CPU, Naive GPU, Work-Efficient GPU

## Preliminary Analysis

### Scan Analysis
