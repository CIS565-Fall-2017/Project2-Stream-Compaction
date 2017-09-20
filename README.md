CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Fengkai Wu
* Tested on: Windows 10, i7-6700 @ 3.40GHz 16GB, Quadro K620 4095MB (Twn M70 Lab)

### Analysis

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

The running time of exclusive scan under different algorithms are as follows:
![img_1](https://github.com/wufk/Project2-Stream-Compaction/blob/master/img/Scan.png)

![img_1](https://github.com/wufk/Project2-Stream-Compaction/blob/master/img/proj2table.PNG)

The running time of stream compaction is as follows:
![img_2](https://github.com/wufk/Project2-Stream-Compaction/blob/master/img/cmpact.png)

As the graph shows, naive scan takes extreme long time to finish the job while the efficient way is much fast. However, the GPU performance is still not as good as CPU. In my Implementation of efficient scan, for each downSweep/upSweep, the number of actual number of working threads is re-computed. The launching blocks are also derived from the number of threads to be used. Bits shifting and modulus operation are also avoided. Other possible factors that downplay the performance might due to too many kernel calls when sweeping up and down, large use of global memory and too many threads required when the array size is large.

Possible ways to further enhance the performance in the future includes using shared memory and dividing  and scanning the array by blocks.

Another worth noticing is that thrust runs way faster when the array size is non multiple of two. 

The timeline of execution when the array size is 2^20 is as follows:
![img_2](https://github.com/wufk/Project2-Stream-Compaction/blob/master/img/proj2Perform.PNG)

It shows that CUDA library of memery manipulation is very expensive. Furthermore, we can see that using thrust and using our own algorithms of scanning is calling different CUDA runtime API. Thrust is calling cudaDeviceSynchronize function while our algorithms call cudaEventSynchronize. This may partly explain why thrust run way faster, in that it is optimized in device and hardware while our effort is just focusing on algorithm and high level part.

In summary, to get better performance in GPU computing, architecture makes a huge difference and optimization must focus on better allocating resources and making use of the specaiality of GPU hardware. 
