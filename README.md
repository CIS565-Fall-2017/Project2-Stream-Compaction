CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

```

****************
** SCAN TESTS **
****************
a[SIZE]:
    [  44  34   5  30  42  18  12  23  14   1  12  32   1 ...   7   0 ]
a[NPOT]:
    [  44  34   5  30  42  18  12  23  14   1  12  32   1 ...  27  45 ]
==== cpu scan, power-of-two ====
   elapsed time: 2.31825ms    (std::chrono Measured)
    [   0  44  78  83 113 155 173 185 208 222 223 235 267 ... 25692019 25692026 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 2.28568ms    (std::chrono Measured)
    [   0  44  78  83 113 155 173 185 208 222 223 235 267 ... 25691912 25691939 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 16.4329ms    (CUDA Measured)
    [   0  44  78  83 113 155 173 185 208 222 223 235 267 ... 25692019 25692026 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 16.4599ms    (CUDA Measured)
    [   0  44  78  83 113 155 173 185 208 222 223 235 267 ... 25691912 25691939 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 2.88461ms    (CUDA Measured)
    [   0  44  78  83 113 155 173 185 208 222 223 235 267 ... 25692019 25692026 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 3.04077ms    (CUDA Measured)
    [   0  44  78  83 113 155 173 185 208 222 223 235 267 ... 25691912 25691939 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.89341ms    (CUDA Measured)
    [   0  44  78  83 113 155 173 185 208 222 223 235 267 ... 25692019 25692026 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.731264ms    (CUDA Measured)
    [   0  44  78  83 113 155 173 185 208 222 223 235 267 ... 25691912 25691939 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   1   2   2   0   2   1   0   1   0   0   1 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.66989ms    (std::chrono Measured)
    [   1   2   2   2   1   1   1   3   2   1   3   2   2 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.62928ms    (std::chrono Measured)
    [   1   2   2   2   1   1   1   3   2   1   3   2   2 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 10.7862ms    (std::chrono Measured)
    [   1   2   2   2   1   1   1   3   2   1   3   2   2 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 3.89808ms    (CUDA Measured)
    [   1   2   2   2   1   1   1   3   2   1   3   2   2 ...   1   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 3.88435ms    (CUDA Measured)
    [   1   2   2   2   1   1   1   3   2   1   3   2   2 ...   3   3 ]
    passed

```
