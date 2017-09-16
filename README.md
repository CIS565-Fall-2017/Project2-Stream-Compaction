**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 2 - Stream Compaction**

* Josh Lawrence
* Tested on: Windows 10, i7-6700HQ @ 2.6GHz 8GB, GTX 960M 2GB  Personal

**CMakeLists.txt Additions**
radix.h<br />
radix.cu<br />
sharedandbank.h<br />
sharedandbank.cu<br />

![](img/ThrustCudaLaunches.png)
![](img/ThrustTimeline.png)
![](img/EfficientTimeline.png)


**GPU Device Properties**
https://devblogs.nvidia.com/parallelforall/5-things-you-should-know-about-new-maxwell-gpu-architecture/<br />
cuda cores 640<br />
mem bandwidth 86.4 GB/s<br />
L2 cache size 2MB<br />
number of multiprocessor 5<br />
max blocks per multiprocessor 32<br />
total shared mem per block 49152 bytes<br />
total shared mem per MP 65536 bytes<br />
total regs per block and MP 65536<br />
max threads per block 1024<br />
max threads per mp 2048<br />
total const memory 65536<br />
max reg per thread 255<br />
max concurrent warps 64<br />
total global mem 2G<br />
<br />
max dims for block 1024 1024 64<br />
max dims for a grid 2,147,483,647 65536 65536<br />
clock rate 1,097,5000<br />
texture alignment 512<br />
concurrent copy and execution yes<br />
major.minor 5.0<br />


**Debug Print**
///SIZE: 16777216<br />
///****************<br />
///** SCAN TESTS **<br />
///****************<br />
///    [  38  40  41  43  11  22  37  25  22  10  11  31  20 ...   6   0 ]<br />
///==== cpu scan, power-of-two ====<br />
///   elapsed time: 68.8818ms    (std::chrono Measured)<br />
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886910 410886916 ]<br />
///==== cpu scan, non-power-of-two ====<br />
///   elapsed time: 68.7083ms    (std::chrono Measured)<br />
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886889 410886895 ]<br />
///    passed<br />
///==== naive scan, power-of-two ====<br />
///   elapsed time: 55.8763ms    (CUDA Measured)<br />
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886910 410886916 ]<br />
///    passed<br />
///==== naive scan, non-power-of-two ====<br />
///   elapsed time: 55.8796ms    (CUDA Measured)<br />
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ...   0   0 ]<br />
///    passed<br />
///==== work-efficient scan, power-of-two ====<br />
///   elapsed time: 24.0157ms    (CUDA Measured)<br />
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886910 410886916 ]<br />
///    passed<br />
///==== work-efficient scan, non-power-of-two ====<br />
///   elapsed time: 24.0113ms    (CUDA Measured)<br />
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886889 410886895 ]<br />
///    passed<br />
///==== thrust scan, power-of-two ====<br />
///   elapsed time: 4.21507ms    (CUDA Measured)<br />
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886910 410886916 ]<br />
///    passed<br />
///==== thrust scan, non-power-of-two ====<br />
///   elapsed time: 4.23005ms    (CUDA Measured)<br />
///    [   0  38  78 119 162 173 195 232 257 279 289 300 331 ... 410886889 410886895 ]<br />
///    passed<br />
///<br />
///*****************************<br />
///** STREAM COMPACTION TESTS **<br />
///*****************************<br />
///    [   1   1   0   0   0   1   3   0   3   2   1   3   1 ...   1   0 ]<br />
///==== cpu compact without scan, power-of-two ====<br />
///   elapsed time: 149.105ms    (std::chrono Measured)<br />
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   1 ]<br />
///    passed<br />
///==== cpu compact without scan, non-power-of-two ====<br />
///   elapsed time: 146.049ms    (std::chrono Measured)<br />
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   3 ]<br />
///    passed<br />
///==== cpu compact with scan ====<br />
///   elapsed time: 369.818ms    (std::chrono Measured)<br />
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   1 ]<br />
///    passed<br />
///==== work-efficient compact, power-of-two ====<br />
///   elapsed time: 214.212ms    (CUDA Measured)<br />
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   1 ]<br />
///    passed<br />
///==== work-efficient compact, non-power-of-two ====<br />
///   elapsed time: 206.909ms    (CUDA Measured)<br />
///    [   1   1   1   3   3   2   1   3   1   1   3   3   2 ...   1   3 ]<br />
///    passed<br />
///Press any key to continue . . .<br />
