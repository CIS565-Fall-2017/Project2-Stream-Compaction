CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Anton Khabbaz
* pennkey:akhabbaz
* Tested on: Windows 10 surface book i7-6600u at 2.66 GHz with a GPU GTX 965M
Personal computer

### (TODO: Your README)


Here I implemented the efficient scan using contiguous threads.  This worked perfectly up to one block but beyond one block the code failed.   The issue was that threads beyond one block do not synchronize.

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

