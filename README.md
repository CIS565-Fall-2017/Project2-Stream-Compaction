CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* YAOYI BAI (pennkey: byaoyi)
* Tested on: Windows 10 Professional, i7-6700HQ  @2.60GHz 16GB, GTX 980M 8253MB (My own Dell Alienware R3)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

# 1. CPU Scan & Stream Compaction #

## Results ##

**1. Scan Test**

![](https://i.imgur.com/hMj3J8b.jpg)

**2. Stream Compaction Test**

![](https://i.imgur.com/r6Fe9QO.jpg)

## Performance ##

**Scan Tests:**

Power-of-two: 0.008692 ms

Non-power-of-two: 0.001185 ms 

**Compaction Tests:**

Without Scan power-of-two: 0.003555 ms

Without scan non-power-of-two: 0.00395 ms

With scan: 0.005531 ms

# 2. Naive GPU Scan Compaction #

## Results ##

**1. Scan Test**

![](https://i.imgur.com/qZd1md3.jpg)

## Performance ##

Naive power-of-two: 1.65587 ms

Naive non-power-of-two: 1.6399 ms

# 3. Work Efficient GPU Scan & Stream Compaction #

## Results ##

## Performance ##

# 4. Thrust's Implementation #

## Results ##

**1. Scan Test**

![](https://i.imgur.com/EHjIGAA.jpg)

## Performance ##

Thrust power-of-two: 30.4844 ms

Thrust non-power-of-two: 1.6409 ms

# 5. Why is my GPU Approach so slow? (extra credits)#

The result of Naive scanning as well as efficient are far more slower than CPU's results. 

## Results ##

## Performance ##

# 7. GPU Scan Using Shared Memory & Hardware Optimization (extra credits) #

## Results ##

## Performance ##