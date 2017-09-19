University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2 CUDA Stream Compaction
======================
* Ziyu Li
* Tested on: Windows 7, i7-3840QM @ 2.8GHz 16GB, Nivida Quadro K4000M 4096MB (Personal Laptop)

## Performance Analysis
#### Efficient Scan with Optimization
To avoid the efficient scan method uses extra non-working threads, simply change the index pattern to perform the kernels. So this optimization can reduce a huge amount of threads to perform useless operations and increase the overall performance.

For the benchmark and result, please check **Performance** section


#### More Efficient Scan with Shared Memory
The optimized method which states above still is not efficiency enough. Actually by performing the operations in shared memory can highly achieve the maximum the performance. The whole implementation can split into three parts.

* Scan each blocks seperatly and use a auxiliary array to store each block sum
* Scan the block sums
* Add scanned block sum to next scanned block

![](img/39fig06.jpg)

(Figure 1: Algorithm for Performing a Sum Scan on a Large Array of Values, Nvidia GPU Gems)


This implementation is relatively easy to achieve, however using share memory will sometimes suffer from bank conflicts which could hurt the performance significantly by access those memory everytime. To avoid these bank conflict, we have to add padding to share memory every certain elements. And those offset can be easily implement by a macro.

```c++
#define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
```

For the benchmark and result, please check **Performance** section


## Performance
#### Scan Performace Measurement and Result
The benckmark is performed the scan operation under 128 threads per block for array size from 2^4 to 2^22. (Since there is only one grid, 2^22 is the maximum amount for a 128 block size.)

The benchmark also makes a running time comparision between CPU, GPU Naive, GPU Efficient, GPU Efficient With Optimization, GPU Efficient With Share Memory and GPU Thrust Scan.

![](img/scan_power_2.PNG)

![](img/scan_power_not_2.PNG)

(For the detail result, please check the data in the **benckmark** folder)

#### Compact Performace Measurement and Result
The benckmark is performed the compaction operation under 128-512 threads per block for array size from 2^4 to 2^24. (128 block size for array size 2^4 to 2^22, 256 block size for 2^23 and 512 block size for 2^24)

The benchmark also makes a running time comparision between CPU without scan, CPU with scan and GPU with scan.

![](img/compaction_2.PNG)


(For the detail result, please check the data in the **benckmark** folder)

## Questions




