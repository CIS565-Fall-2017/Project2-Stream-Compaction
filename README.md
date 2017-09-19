CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Carolina Zheng
* Tested on: Windows 7, i7-6700 @ 3.40GHz 16GB, Quadro K620 (Moore 100 Lab)

### Performance Analysis

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

#### Test Program Output
```
****************
** SCAN TESTS **
****************
    [   1  42  48  47   9  36  13  34  28  39   3   2  38 ...   1   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 2.4675ms    (std::chrono Measured)
==== cpu scan, non-power-of-two ====
   elapsed time: 2.47652ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
   elapsed time: 38.5892ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 37.0342ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 6.30717ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 6.17421ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 84.909ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 78.5352ms    (CUDA Measured)
    passed


*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   0   3   1   0   1   3   1   0   0   2   0 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 4.67753ms    (std::chrono Measured)
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 4.78749ms    (std::chrono Measured)
    passed
==== cpu compact with scan ====
   elapsed time: 12.9883ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 13.6215ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 13.4029ms    (CUDA Measured)
    passed
Press any key to continue . . .
```
