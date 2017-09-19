CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* LINSHEN XIAO
* Tested on: Windows 10, Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz, 16.0GB, NVIDIA GeForce GTX 970M (Personal computer)

## Features

* CPU Scan
* CPU Stream Compaction
* Naive GPU Scan Algorithm
* Work-Efficient GPU Scan
* Work-Efficient GPU Stream Compaction
* Thrust's Implementation

## Performance analysis

* Comparision of GPU Scan(Naive, Work-Efficient、) for different block size.

| Block   size | naive scan,   power-of-two | naive scan,   non-power-of-two | work-efficient   scan, power-of-two | work-efficient   scan, non-power-of-two |
|--------------|----------------------------|--------------------------------|-------------------------------------|-----------------------------------------|
| 32           | 2.31232                    | 2.30061                        | 1.0649                              | 1.03421                                 |
| 64           | 1.75728                    | 1.75683                        | 1.02925                             | 1.06506                                 |
| 128          | 1.76538                    | 1.77981                        | 1.02992                             | 1.04938                                 |
| 256          | 1.79238                    | 1.76445                        | 1.03142                             | 1.03942                                 |
| 512          | 1.75421                    | 1.75085                        | 1.10819                             | 1.03946                                 |
| 1024         | 1.75885                    | 1.75379                        | 1.06323                             | 1.05293                                 |

* Comparision of GPU Scan(Naive, Work-Efficient & Thrust) and CPU version.

| Array size | cpu scan, power-of-two | cpu scan, non-power-of-two | naive scan, power-of-two | naive scan, non-power-of-two | work-efficient scan, power-of-two | work-efficient scan, non-power-of-two | thrust scan, power-of-two | thrust scan, non-power-of-two |
|------------|------------------------|----------------------------|--------------------------|------------------------------|-----------------------------------|---------------------------------------|---------------------------|-------------------------------|
| 2^8        | 0.00079                | 0.000395                   | 0.046112                 | 0.035296                     | 0.106176                          | 0.074688                              | 0.03664                   | 0.013728                      |
| 2^10       | 0.001975               | 0.001975                   | 0.047552                 | 0.055584                     | 0.106112                          | 0.098848                              | 0.023136                  | 0.013696                      |
| 2^12       | 0.007111               | 0.007506                   | 0.067552                 | 0.065088                     | 0.120032                          | 0.182144                              | 0.029984                  | 0.020512                      |
| 2^14       | 0.030419               | 0.061234                   | 0.074976                 | 0.064032                     | 0.145696                          | 0.140928                              | 0.05952                   | 0.047424                      |
| 2^16       | 0.118124               | 0.120889                   | 0.116256                 | 0.115424                     | 0.185984                          | 0.18752                               | 0.153856                  | 0.163648                      |
| 2^18       | 1.34716                | 0.503309                   | 0.408064                 | 0.397536                     | 0.313952                          | 0.298272                              | 0.228928                  | 0.36768                       |
| 2^20       | 2.25462                | 2.55684                    | 1.7671                   | 1.75427                      | 1.08269                           | 1.07862                               | 0.310208                  | 0.406784                      |
| 2^22       | 8.39032                | 8.32909                    | 7.86064                  | 7.81309                      | 3.96986                           | 3.93744                               | 0.865472                  | 0.864512                      |
| 2^24       | 37.5455                | 35.584                     | 34.8642                  | 34.8712                      | 15.5057                           | 15.5434                               | 2.62048                   | 2.70346                       |

* Comparision of GPU Stream compaction(Naive, Work-Efficient & Thrust) and CPU version.

| Array size | cpu compact without scan, power-of-two | cpu compact without scan, non-power-of-two | cpu compact with scan | work-efficient compact, power-of-two | work-efficient compact, non-power-of-two |
|------------|----------------------------------------|--------------------------------------------|-----------------------|--------------------------------------|------------------------------------------|
| 2^8        | 0.001185                               | 0.001186                                   | 0.003951              | 0.094752                             | 0.093824                                 |
| 2^10       | 0.00395                                | 0.00316                                    | 0.024493              | 0.176832                             | 0.10768                                  |
| 2^12       | 0.011851                               | 0.012247                                   | 0.076246              | 0.174656                             | 0.202976                                 |
| 2^14       | 0.047407                               | 0.063605                                   | 0.126025              | 0.20528                              | 0.160288                                 |
| 2^16       | 0.193975                               | 0.185284                                   | 0.463013              | 0.211744                             | 0.243264                                 |
| 2^18       | 1.04138                                | 0.776692                                   | 2.61413               | 0.368416                             | 0.360832                                 |
| 2^20       | 3.66025                                | 3.52553                                    | 8.02608               | 1.46659                              | 1.40083                                  |
| 2^22       | 18.9922                                | 12.979                                     | 39.1909               | 5.3247                               | 5.31274                                  |
| 2^24       | 50.8401                                | 57.8019                                    | 135.726               | 20.4587                              | 20.4935                                  |

## Questions

* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.

  * (You shouldn't compare unoptimized implementations to each other!)

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).

  * We wrapped up both CPU and GPU timing functions as a performance timer class for you to conveniently measure the time cost.
    * We use `std::chrono` to provide CPU high-precision timing and CUDA event to measure the CUDA performance.
    * For CPU, put your CPU code between `timer().startCpuTimer()` and `timer().endCpuTimer()`.
    * For GPU, put your CUDA code between `timer().startGpuTimer()` and `timer().endGpuTimer()`. Be sure **not** to include any *initial/final* memory operations (`cudaMalloc`, `cudaMemcpy`) in your performance measurements, for comparability.
    * Don't mix up `CpuTimer` and `GpuTimer`.
  * To guess at what might be happening inside the Thrust implementation (e.g.
    allocation, memory copy), take a look at the Nsight timeline for its
    execution. Your analysis here doesn't have to be detailed, since you aren't
    even looking at the code for the implementation.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

These questions should help guide you in performance analysis on future
assignments, as well.

