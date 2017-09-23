CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Xincheng Zhang
* Tested on: (TODO) Windows 10, i7-4702HQ @ 2.20GHz 8GB, GTX 870M 3072MB (Personal Laptop)

### Output Screenshot
-------------
@blocksize = 128; Arraysize = 1<<9
![](https://github.com/XinCastle/Project2-Stream-Compaction/blob/master/img/sc1.png)
![](https://github.com/XinCastle/Project2-Stream-Compaction/blob/master/img/sc2.png)
![](https://github.com/XinCastle/Project2-Stream-Compaction/blob/master/img/sc3.png)

### Description&Features
-------------
```
1: CPU Scan; Stream Compaction
2: Naive Scan using GPU
3: Efficient GPU Scan; Stream Compaction
4: Thrust Scan
```

### Blocksize Optimization
-------------
@constant Arraysize = 1<<9, the performance of different methods will change accroding to the blocksize. Therefore, I modify the blocksize to find the optimized value of these methods.

![](https://github.com/XinCastle/Project2-Stream-Compaction/blob/master/img/chart1.png)

**The test data of the chart above is the following:**
Block Size | Naive Scan | Efficient Scan | Thrust Scan
---|---|---|---
32 | 0.3818  | 0.1598 |1.0674
64 | 0.0389  | 0.1575 |1.0808
128 | 0.0382 | 0.1373 |1.0888
256 | 0.0387 | 0.1542 |1.0669
512 | 0.0428 | 0.1398 |1.0899
1024 | 0.043 | 0.1532 |1.0523

From the data I get and the chart above, we can tell that for CPU scan, the blocksize doesn't change the performance. For naive scan, its best blocksize is 128. For efficient scan, its best blocksize is 128. As for thrust scan, its best blocksize is 1024.


### Performance Comparison Based on Array Size
-------------
Array Size | Naive Scan | Efficient Scan | Thrust Scan | CPU Scan
---|---|---|---|---
2^8 | 0.3546  | 0.127 |1.0821 |0.0014
2^12 | 0.0531  | 0.1795 |2.3651 |0.1398
2^16 | 0.2922 | 0.6992 |8.2656 |0.265
2^20 | 3.3498 | 7.4632 |40.6058 |3.2167
2^24 | 61.6843 | 130.091 |556.343 |53.3077

The chart is the following:
![](https://github.com/XinCastle/Project2-Stream-Compaction/blob/master/img/chart2.png)


### Questions
-------------
*  Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.
Answer: in the "blocksize optimization" above.

* Compare all of these GPU Scan implementations
Answer: in the "Performance Comparison Based on Array Size" above. I guess that thrust scan uses shared memory.

* Write a brief explanation of the phenomena you see here.
Answer: I think the reason why GPU methods are slower than CPU method is because that in these methods, not all the threads are working which means we have lots of threads doing nothing so they are not efficient enough to be faster than CPU scan. Moreover, I think I/O is another factor that causes bottleneck because there are many memory copy operations in my code.


