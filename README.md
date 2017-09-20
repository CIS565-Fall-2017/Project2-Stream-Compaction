CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2 - Stream Compaction**

* Joseph Klinger
* Tested on: Windows 10, i5-7300HQ (4 CPUs) @ ~2.50GHz, GTX 1050 6030MB (Personal Machine)

### README

This project consists of a series of implementations of the inclusive/exclusive scan and stream compaction across the CPU and GPU.
We implement a sequential CPU scan, a non-work-efficient GPU scan (naive), and an actually work-efficient GPU scan.

A scan is a prefix sum (assuming an array of integers for now), meaning that the index i in the output array will consist of the sum of each previous element
in the input array. Here's a concrete example of each kind of scan, taken from the slides by Patrick Cozzi and Shehzan Mohammed [here](https://docs.google.com/presentation/d/1ETVONA7QDM-WqsEj4qVOGD6Kura5I6E9yqH-7krnwZ0/edit#slide=id.p27)

![](img/scans.png)

Stream Compaction utilizes scans as a way to reduce an array of any integers to an array consisting only of the integers that meet a certain criteria (say, not equal to 0).
Taking some images from the same slides as before, here is step 1 to stream compaction - create an array of 1s and 0s indicating whether or not we want to keep a certain element, then
run an exclusive scan on that array:

![](img/compact.png)

So, in this example, we want to keep elements a, c, d, and g.

As it turns out, for each element in the input array that has a 1 in the intermediate array, the corresponding value in the summed array is that element's index in the final output array.
This step is called scatter:

![](img/compact2.png)

Now we are left with our desired array.

### Analysis
Here are the analysis results from my implementations:

![](img/graphAllScans.png)

Here we can get a general vibe that the naive parallel scan didn't work so well, the CPU scan performed decently, and adding the work-efficient implementation is a must, and thrust wins easily.

One major aspect to note is the difference between the two work-efficient parallel scans. One utilizes a naive arrangement of block launching and the other is more intelligent (only launching
threads as needed). In the following graph, we can see that this optimization is what allows the parallel implementation to outperform the CPU implementation:

![](img/graphGPUandCPU.png)

Now the work-efficient parallel scan is good, but still, the 3rd party Thrust implementation still by far outperforms all other implementations (as you can see in the first graph too):

![](graphGPU.png)

There are further optimizations that can be added to the work-efficient parallel scan that can help it compete with Thrust, such as utilizing shared memory, but they weren't completed for this project (yet!).
