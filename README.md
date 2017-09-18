CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Byumjin Kim
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 15.89GB (Personal labtop)

### Result

![](img/Result.png)

Result with 2^22 elements

### Complete Specs

- Part 1: CPU Scan & Stream Compaction
- Part 2: Naive GPU Scan Algorithm
- Part 3: Work-Efficient GPU Scan & Stream Compaction
- Part 4: Using Thrust's Implementation
- Part 5: Why is My GPU Approach So Slow?
- Part 6: Radix Sort
- Part 7: GPU Scan Using Shared Memory

### Performance Analysis

## Scan performance (ms) depending on the array size

For readability, below graph only shows power of two version of each implementation methods except Thrust.

![](img/01.png)

| # of Elements  				| 2^8	   | 2^9	  | 2^10	 | 2^11		| 2^12	   | 2^13     | 2^14     | 2^15     | 2^16     | 2^17     | 2^18     | 2^19     | 2^20     | 2^21     | 2^22     |
| ------------- 				| :------- | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | -------: |
| CPU Scan (power of two)		| 0.001185 | 0.001185 | 0.003160 | 0.003951 | 0.008296 | 0.017382 | 0.033975 | 0.082172 | 0.133136 | 0.282070 | 0.614715 | 1.246020 | 2.875260 | 5.215600 | 11.30860 |
| CPU Scan (non power of two)	| 0.000395 | 0.001185 | 0.001967 | 0.003951 | 0.007901 | 0.015803 | 0.031605 | 0.066765 | 0.128790 | 0.257580 | 0.783011 | 1.291450 | 3.154560 | 5.071400 | 11.09170 |
| Naïve Scan (power of two)		| 0.025248 | 0.031296 | 0.044512 | 0.051008 | 0.054720 | 0.045568 | 0.062688 | 0.066240 | 0.083552 | 0.121664 | 0.324160 | 0.621920 | 1.251330 | 2.605600 | 5.466850 |
| Naïve Scan (non power of two)	| 0.028032 | 0.031200 | 0.037120 | 0.050208 | 0.054432 | 0.048448 | 0.051264 | 0.063232 | 0.085088 | 0.169984 | 0.342336 | 0.618688 | 1.248450 | 2.610530 | 5.461220 |
| Efficient (power of two)		| 0.011296 | 0.013312 | 0.014560 | 0.015648 | 0.021088 | 0.020448 | 0.021504 | 0.025664 | 0.040672 | 0.050048 | 0.085312 | 0.140560 | 0.282080 | 0.543744 | 1.070240 |
| Efficient (non power of two)	| 0.022816 | 0.012768 | 0.013920 | 0.014784 | 0.020544 | 0.019712 | 0.022592 | 0.024640 | 0.035488 | 0.049984 | 0.083616 | 0.150816 | 0.281536 | 0.540128 | 1.064030 |
| Thrust (power of two)			| 4.762620 | 4.991040 | 4.287230 | 5.425380 | 5.470780 | 4.596580 | 5.378050 | 5.273600 | 5.203870 | 5.178370 | 5.251970 | 5.058560 | 5.350400 | 5.319580 | 5.170180 |
| Thrust (non power of two)		| 0.012896 | 0.012576 | 0.012960 | 0.015328 | 0.017760 | 0.026752 | 0.041568 | 0.425888 | 0.222080 | 0.234208 | 0.212960 | 0.273408 | 0.305792 | 0.438272 | 0.758464 |

When I implemented my Naïve and Efficient scan functions for the first time, I didn't use shared memory. But, after I used it for both functions, it gave almost 10 times faster performance than before. (The above results are from the functions using shared memory)
But, even if I used shared memory, my Naïve and Efficient scan functions couldn't be faster than CPU scan at the cases which used relatively small number of elements.
When the number of elements increased over certain digits(2^14), the Efficient Scan beated the those of CPU. Because, more elements give more efficient environment for using parrallel programming.

The Efficient Scan also beated Thrust Scan. But, after using 2^20 elements, its performance was slower than those of Thrust again.
It seems Thrust implemenataion's memory allocation, copy, block size and etc are optimized better than those of my Efficient Scan.
And, I refered to GPU Gem3's "39.2.4 Arrays of Arbitrary Size" for implementing my Efficient Scan to remove the array size limitation like my Naïve Scan.
So, when my Efficient Scan tries to compact its result from a large array of values to a small array, it is possible to make some bottle-necks. 

An interesting point was, it was really difficult to check the beneficial of using power of two elements.
And, another one was, power of two version of Thrust implementation's performance was too expensive what I expected.

## Stream Compaction performance (ms) depending on the array size

![](img/02.png)

| # of Elements  								| 2^8	   | 2^9	  | 2^10	 | 2^11		| 2^12	   | 2^13	  | 2^14	 | 2^15     | 2^16     | 2^17     | 2^18     | 2^19	    | 2^20     |
| ------------- 								| :------- | :------: | -------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | -------: |
| CPU Compaction w/o Scan (power of two)		| 0.000790 | 0.001580 | 0.003555 | 0.010666 | 0.021334 | 0.076642 | 0.092444 | 0.248098 | 0.400988 | 0.713876 | 1.603950 | 3.467460 | 6.605430 |
| CPU Compaction w/o Scan (non power of two)	| 0.001185 | 0.001581 | 0.003555 | 0.011061 | 0.024099 | 0.079802 | 0.086914 | 0.222814 | 0.514765 | 0.789728 | 1.522170 | 2.827850 | 5.806220 |
| CPU Compaction with Scan						| 0.001976 | 0.003555 | 0.007506 | 0.027654 | 0.055309 | 0.141432 | 0.332642 | 0.517136 | 0.880592 | 2.403160 | 3.781920 | 9.545870 | 18.00220 |
| Efficient (power of two)						| 0.018592 | 0.020800 | 0.020864 | 0.024768 | 0.028896 | 0.029760 | 0.031744 | 0.036128 | 0.058560 | 0.075584 | 0.163904 | 0.271008 | 0.510944 |
| Efficient (non power of two)					| 0.020576 | 0.021856 | 0.023616 | 0.024576 | 0.026944 | 0.029696 | 0.049184 | 0.046144 | 0.054048 | 0.075616 | 0.141792 | 0.272544 | 0.511008 |

The Efficient implementation is more efficient with huge number of elements (from 2^13).

## Radix Sort performance (ms) depending on the array size

![](img/03.png)

| # of Elements  | 2^8	    | 2^9	   | 2^10     | 2^11     | 2^12     | 2^13     | 2^14     | 2^15     | 2^16     | 2^17     | 
| -------------  | :------- | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | -------: |
| Standard Sort	 | 0.024494 | 0.035950 | 0.061630 | 0.135506 | 0.250470 | 0.597728 | 1.063110 | 2.229330 | 6.077230 | 10.37950 |
| Radix Sort	 | 0.154976 | 0.188832 | 0.230400 | 0.266144 | 0.354208 | 0.434784 | 0.498432 | 0.590144 | 0.867200 | 1.597380 |

The Radix Sort implementation is more efficient with huge number of elements (from 2^13).
