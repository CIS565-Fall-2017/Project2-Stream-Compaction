CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Sarah Forcier
* Tested on GeForce GTX 1070

### Description
#### * Scan
#### * Stream Compaction
#### * Radix Sort

### Test Output

#### * Scan

```
    [  15  26  19   6  48  18   4  40   8  13  32  26  32  37  14   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.000277ms    (std::chrono Measured)
    [   0  15  41  60  66 114 132 136 176 184 197 229 255 287 324 338 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.000277ms    (std::chrono Measured)
    [   0  15  41  60  66 114 132 136 176 184 197 229 255 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.012288ms    (CUDA Measured)
    [   0  15  41  60  66 114 132 136 176 184 197 229 255 287 324 338 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.010496ms    (CUDA Measured)
    [   0  15  41  60  66 114 132 136 176 184 197 229 255 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.017664ms    (CUDA Measured)
    [   0  15  41  60  66 114 132 136 176 184 197 229 255 287 324 338 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.017472ms    (CUDA Measured)
    [   0  15  41  60  66 114 132 136 176 184 197 229 255 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 10.5801ms    (CUDA Measured)
    [   0  15  41  60  66 114 132 136 176 184 197 229 255 287 324 338 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.014368ms    (CUDA Measured)
    [   0  15  41  60  66 114 132 136 176 184 197 229 255 ]
    passed
```

#### * Stream Compaction

```
    [   3   1   1   1   2   1   1   0   3   1   0   2   3   0   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.000277ms    (std::chrono Measured)
    [   3   1   1   1   2   1   1   3   1   2   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.000277ms    (std::chrono Measured)
    [   3   1   1   1   2   1   1   3   1   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.001387ms    (std::chrono Measured)
    [   3   1   1   1   2   1   1   3   1   2   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.216224ms    (CUDA Measured)
    [   3   1   1   1   2   1   1   3   1   2   3   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.099584ms    (CUDA Measured)
    [   3   1   1   1   2   1   1   3   1   2   3 ]
    passed
```

#### * Radix Sort

```
    [  21  23  29  15  10  21  29  36  13  17  36  14  41  12  31   0 ]
==== cpu sort, power-of-two ====
   elapsed time: 0.000554ms    (std::chrono Measured)
    [   0  10  12  13  14  15  17  21  21  23  29  29  31  36  36  41 ]
==== cpu sort, non-power-of-two ====
   elapsed time: 0.000555ms    (std::chrono Measured)
    [  10  13  14  15  17  21  21  23  29  29  36  36  41 ]
==== radix sort, power-of-two ====
   elapsed time: 0.099584ms    (CUDA Measured)
    [   0  10  12  13  14  15  17  21  21  23  29  29  31  36  36  41 ]
    passed
==== radix sort, non-power-of-two ====
   elapsed time: 0.099584ms    (CUDA Measured)
    [  10  13  14  15  17  21  21  23  29  29  36  36  41 ]
    passed
 ```

### Performance Analysis

### Q&A

#### Compare GPU Scan implementations to the serial CPU version?

#### Where are the performance bottlenecks?
