CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**
Ju Yang 
### Tested on: Windows 7, i7-4710MQ @ 2.50GHz 8GB, GTX 870M 6870MB (Hasee Notebook K770E-i7)
![result](doc/1024.png)

## TODOs finished: 
  ### 1. naive.cu 
  __global__ void naive_sum(int n,int* odata, int* idata)
  void scan(int n, int *odata, const int *idata)
  
  ### 2. efficient.cu 
  __global__ void prescan(int *g_odata, int *g_idata, int n, int*temp)
  void scan(int n, int *odata, const int *idata)
  int compact(int n, int *odata, const int *idata)

  ### 3 thrust.cu 
  void scan(int n, int *odata, const int *idata)
  
  ### 4 cpu.cu 
  void scan(int n, int *odata, const int *idata)
  int compactWithoutScan(int n, int *odata, const int *idata) 
  int compactWithScan(int n, int *odata, const int *idata)
  
  ### 5 common.cu 
  __global__ void kernMapToBoolean(int n, int *bools, const int *idata)
  __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices)
                
 ### Modified the main.cpp a little bit for display. 

