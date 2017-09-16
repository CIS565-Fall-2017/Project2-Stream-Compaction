**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 2 - Stream Compaction**

* Josh Lawrence
* Tested on: Windows 10, i7-6700HQ @ 2.6GHz 8GB, GTX 960M 2GB  Personal

** CMakeLists.txt Additions**
radix.h
radix.cu
sharedandbank.h
sharedandbank.cu

![](img/ThrustCudaLaunches.png)
![](img/ThrustTimeline.png)
![](img/EfficientTimeline.png)


**GPU Device Properties**
https://devblogs.nvidia.com/parallelforall/5-things-you-should-know-about-new-maxwell-gpu-architecture/
cuda cores 640
mem bandwidth 86.4 GB/s
L2 cache size 2MB
number of multiprocessor 5
max blocks per multiprocessor 32
total shared mem per block 49152 bytes
total shared mem per MP 65536 bytes
total regs per block and MP 65536
max threads per block 1024
max threads per mp 2048
total const memory 65536
max reg per thread 255
max concurrent warps 64
total global mem 2G

max dims for block 1024 1024 64
max dims for a grid 2,147,483,647 65536 65536
clock rate 1,097,5000
texture alignment 512
concurrent copy and execution yes
major.minor 5.0


**Debug Print**
///SIZE: 16777216
///****************
///** SCAN TESTS **
///****************
///    [  38  40  41  43  11  22  37  25  22  10  11  31  20 ...   6   0 ]
///==== cpu scan, power-of-two ====
///   elapsed time: 68.8818ms    (std::chrono Measured)
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886910 410886916 ]
///==== cpu scan, non-power-of-two ====
///   elapsed time: 68.7083ms    (std::chrono Measured)
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886889 410886895 ]
///    passed
///==== naive scan, power-of-two ====
///   elapsed time: 55.8763ms    (CUDA Measured)
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886910 410886916 ]
///    passed
///==== naive scan, non-power-of-two ====
///   elapsed time: 55.8796ms    (CUDA Measured)
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ...   0   0 ]
///    passed
///==== work-efficient scan, power-of-two ====
///   elapsed time: 24.0157ms    (CUDA Measured)
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886910 410886916 ]
///    passed
///==== work-efficient scan, non-power-of-two ====
///   elapsed time: 24.0113ms    (CUDA Measured)
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886889 410886895 ]
///    passed
///==== thrust scan, power-of-two ====
///   elapsed time: 4.21507ms    (CUDA Measured)
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886910 410886916 ]
///    passed
///==== thrust scan, non-power-of-two ====
///   elapsed time: 4.23005ms    (CUDA Measured)
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886889 410886895 ]
///    passed
///
///*****************************
///** STREAM COMPACTION TESTS **
///*****************************
///    [   1   1   0   0   0   1   3   0   3   2   1   3   1 ...   1   0 ]
///==== cpu compact without scan, power-of-two ====
///   elapsed time: 149.105ms    (std::chrono Measured)
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   1 ]
///    passed
///==== cpu compact without scan, non-power-of-two ====
///   elapsed time: 146.049ms    (std::chrono Measured)
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   3 ]
///    passed
///==== cpu compact with scan ====
///   elapsed time: 369.818ms    (std::chrono Measured)
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   1 ]
///    passed
///==== work-efficient compact, power-of-two ====
///   elapsed time: 214.212ms    (CUDA Measured)
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   1 ]
///    passed
///==== work-efficient compact, non-power-of-two ====
///   elapsed time: 206.909ms    (CUDA Measured)
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   3 ]
///    passed
///Press any key to continue . . .
