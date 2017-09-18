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

![](https://i.imgur.com/o2k1uvp.jpg)

**2. Stream Compaction Test**

![](https://i.imgur.com/r6Fe9QO.jpg)

## Performance ##

**Scan Tests:**

CPU scan Power-of-two: 0.00158 ms

CPU scan Non-power-of-two: 0.001185 ms 

**Compaction Tests:**

CPU compact Without Scan power-of-two: 0.003555 ms

CPU compact Without scan non-power-of-two: 0.00395 ms

CPU compact With scan: 0.005531 ms

# 2. Naive GPU Scan Compaction #

## Results ##

**1. Scan Test**

![](https://i.imgur.com/qZd1md3.jpg)

## Performance ##

Naive power-of-two: 1.65587 ms

Naive non-power-of-two: 1.6399 ms

***Analysis: I think the main reason for GPU implementation to be slower than CPU would be:***

1. The if-else in kernel function. Since we cannot guarantee that all the threads in a warp will go into the same if or else. And it is extremely possible that in one warp, half of the threads in a warp will go into if, and the other half will go into else. Therefore, the performance will be reduced greatly.
2. Data transfer from host to device, from device to device and also from device to host. I think this takes a lot of time than simple calculation inside GPU.
3. Data synchronize is also a problem. In my opinion, I added cudaDeviceSynchronize() and also cudaThreadSynchronize() in host code to avoid data conflict. Although it is not absolutely necessary to put them in the host's code, we need it to keep data correct. However, which means that host and threads in device will wait for each other, which takes time.

# 3. Work Efficient GPU Scan & Stream Compaction #

***Notice: In this part, I did not use the kernels in Common.cu, where I stated my own kernels in Efficient.cu.***

## Results ##

**Scan Tests:**

![](https://i.imgur.com/d78bMWF.jpg)

**Compaction Tests:**

![](https://i.imgur.com/RtMN70r.jpg)

## Performance ##

Efficient scan power-of-two： 1.89722 ms

Efficient scan non-power-of-two: 1.46947 ms

Efficient compact power-of-two: 1.90029 ms

Efficient compact non-power-of-two: 2.48717 ms

***Analysis:***

The algorithm down sweep of non-power-of-two is totally different with the power-of-two. The pseudo code can be illustrated as:

Here an extra array will be included, and the size of this array would be 

    pow(2,ilogceil(n)) - n

This array would be initial set to be a zeros array. And if index + pow(2,d+1) exceeds the limit of original array, it will take the value of this extra array. And write the added up value into this array. However, the array will be freed without any other operation at last.

    if ((index%pow(2,d+1) == 0) OR (index == 0))
			{
				if ((index + pow(2,d) - 1) > n - 1)
				{
					int extraIndex2 = index + pow(2,d+1) - 1 - n;
					int extraIndex1 = index + pow(2,d) - 1 - n;
					int t = extraArray[extraIndex1];
					extraArray[extraIndex1] = extraArray[extraIndex2];
					extraArray[extraIndex2] += t;
				}
				else if((index + pow(2,d+1) - 1) > n - 1)
				{
					int extraIndex = index + pow(2,d+1) - 1 - n;
					int t = idata[index + pow(2,d) - 1];
					idata[index + pow(2,d) - 1] = extraArray[extraIndex];
					extraArray[extraIndex] += t;
				}
				else
				{
					int t = idata[index + pow(2,d) - 1];
					idata[index + pow(2,d) - 1] = idata[index + pow(2,d+1) - 1];
					idata[index + pow(2,d+1) - 1] += t;
				}
			}


However, if-else if-else structure will greatly slow down the performance of GPU, especially when some threads are actually not working at all. To speed up the performance, I think in CPU part of the code, special works that take out the indexes needs to be process should be down before entering the GPU kernels. 

# 4. Thrust's Implementation #

## Results ##

**1. Scan Test**

![](https://i.imgur.com/EHjIGAA.jpg)

## Performance ##

Thrust scan power-of-two: 30.4844 ms

Thrust scan non-power-of-two: 1.6409 ms

# 5. Performance Graphs #

Scan Time

![](https://i.imgur.com/V3P0XML.jpg)

Compact Time

![](https://i.imgur.com/YFaYLeY.jpg)

CPU performance 

![](https://i.imgur.com/5pcXr9z.jpg)

GPU performance (without thrust)

![](https://i.imgur.com/5zZoebj.jpg)