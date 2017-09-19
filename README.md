CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yuxin Hu
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 8GB, GTX 960M 4096MB (Personal Laptop)

### README
#####Project Description
This project is about doing inclusive and exclusive scan for compaction computation in GPU parallel methodology, that will be useful for lots of applications, such as terminating rays that don't contribute in patch tracing.

#####Features Implemented
1. Exclusive Scan
   *CPU Version using single for loop
   *GPU Version using Naive Parallel Reduction
   *GPU Version using Work Efficient Parallel Reduction with a up sweep and down sweep process
   *thrust Version using built in thrust::exclusive_scan function
   
2. Stream Compaction
   *CPU Version of a single for loop without using exclusive scan
   *CPU Version of two for loops with exclusive scan. First for loop scan over the array to check if each element we would like to keep. If yes, we put 1 in a boolean array, if not, we put 0 in a boolean array. Then we call exclusive scan function we wrote in step above to get another array containing exclusive sum of the boolean array. Lastly we use another for loop to get final compaction result from the boolean array and boolean exclusive sum array.
   *GPU Version with work efficient scan. The idea is similar to CPU version, except that the boolean array calculation, exclusive scan, and final result is all done parallelly on GPU.

#####Performance Analysis: 
1. Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.
![BlockSize Versus Efficiency](/img/BlocksizeAndEfficiency.PNG)
<p align="center"><b>BlockSize Versus Efficiency</b></p>

There is not much performance change with block size changes. I set it to 512 for all remaining performance analysis.

2. Scan performance comparason with array size changes
![ScanPerformanceAnalysis](/img/ScanPerformanceAnalysis.PNG)
<p align="center"><b>Exclusive Scan Performance Analysis with Increasing Array Size</b></p>
*It can be observed from the graph that the performance ranking is as follows: thrust > CPU > GPU Naive Scan > GPU Work Efficient Scan. The performance of GPU version scanning is worse than CPU version in my implementation. 
*For GPU Naive Scan, it runs log2n levels, with each level of n threads, so the total number of threads is n*Log2n, which is more than the number of elements check in CPU. Moreover I need to do a right shift of the Naive scan result which is another n threads. Although these threads can run in parallel, but considering the the thread scheduling on GPU, the advantage of naive scan is not that obvious comparing to CPU version. 
*For GPU Work Efficient Scan, I am doint an up sweep and a down sweep, which result in 2*log2n levels of kernal function calls. Each level consists of n threads, so the total number of threads would be 2n*log2n. It needs twice the thread number as Naive scan. In fact, many threads in each level are not doing work because they do not meet index%2^(level+1), but since all the threads in the same warp need to wait for each other to complete their tasks together, the non-functioning threads still takes time.
*thrust's performance is the best of all scan methods.


#####Program Output at array SIZE = 2^15
```
****************
** SCAN TESTS **
****************
    [  44  14  14   1   0  15  26   2  38  20  24  46  10 ...  17   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.061234ms    (std::chrono Measured)
    [   0  44  58  72  73  73  88 114 116 154 174 198 244 ... 801091 801108 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.06084ms    (std::chrono Measured)
    [   0  44  58  72  73  73  88 114 116 154 174 198 244 ... 800978 801016 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.090464ms    (CUDA Measured)
    [   0  44  58  72  73  73  88 114 116 154 174 198 244 ... 801091 801108 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.101856ms    (CUDA Measured)
    [   0  44  58  72  73  73  88 114 116 154 174 198 244 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.222976ms    (CUDA Measured)
    [   0  44  58  72  73  73  88 114 116 154 174 198 244 ... 801091 801108 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.227488ms    (CUDA Measured)
    [   0  44  58  72  73  73  88 114 116 154 174 198 244 ... 800978 801016 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.001184ms    (CUDA Measured)
    [   0  44  58  72  73  73  88 114 116 154 174 198 244 ... 801091 801108 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.001152ms    (CUDA Measured)
    [   0  44  58  72  73  73  88 114 116 154 174 198 244 ... 800978 801016 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   2   0   3   0   3   2   2   0   0   2   0   2 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.12721ms    (std::chrono Measured)
    [   2   2   3   3   2   2   2   2   1   1   2   2   1 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.136691ms    (std::chrono Measured)
    [   2   2   3   3   2   2   2   2   1   1   2   2   1 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.343704ms    (std::chrono Measured)
    [   2   2   3   3   2   2   2   2   1   1   2   2   1 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.403328ms    (CUDA Measured)
    [   2   2   3   3   2   2   2   2   1   1   2   2   1 ...   2   3 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.392096ms    (CUDA Measured)
    [   2   2   3   3   2   2   2   2   1   1   2   2   1 ...   2   3 ]
    passed
Press any key to continue . . .
```


