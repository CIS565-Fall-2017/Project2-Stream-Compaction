CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Name: Meghana Seshadri
* Tested on: Windows 10, i7-4870HQ @ 2.50GHz 16GB, GeForce GT 750M 2048MB (personal computer)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)








****************
** SCAN TESTS **
****************
    [  40  12   3  21  27   2  32  20   1   4  41  34  15 ...  34   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.000821ms    (std::chrono Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6112 6146 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.000821ms    (std::chrono Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6063 6064 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.44064ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6112 6146 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.433024ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6063 6064 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.781184ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6112 6146 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 1.20867ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6099 6099 ]
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6063 6064 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.34442ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6112 6146 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.019392ms    (CUDA Measured)
    [   0  40  52  55  76 103 105 137 157 158 162 203 237 ... 6063 6064 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   0   1   3   1   0   0   2   1   0   1   0   1 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.000821ms    (std::chrono Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.000821ms    (std::chrono Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.962322ms    (std::chrono Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 2.22262ms    (CUDA Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   1   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 2.10656ms    (CUDA Measured)
    [   2   1   3   1   2   1   1   1   3   3   3   3   3 ...   3   1 ]
    passed
Press any key to continue . . .