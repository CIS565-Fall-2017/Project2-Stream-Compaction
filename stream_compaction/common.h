#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}

namespace StreamCompaction {
    namespace Common {
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices);

	    /**
	    * This class is used for timing the performance
	    * Uncopyable and unmovable
        *
        * Adapted from WindyDarian(https://github.com/WindyDarian)
	    */
	    class PerformanceTimer
	    {
	    public:
		    PerformanceTimer()
		    {
			    cudaEventCreate(&event_start);				
			    cudaEventCreate(&event_end);

				cudaEventCreate(&event_start2);
				cudaEventCreate(&event_end2);

				cudaEventCreate(&event_pause_start);
				cudaEventCreate(&event_pause_end);
		    }

		    ~PerformanceTimer()
		    {
			    cudaEventDestroy(event_start);				
			    cudaEventDestroy(event_end);

				cudaEventDestroy(event_start2);
				cudaEventDestroy(event_end2);

				cudaEventDestroy(event_pause_start);
				cudaEventDestroy(event_pause_end);
		    }

		    void startCpuTimer()
		    {
			    if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
			    cpu_timer_started = true;

			    time_start_cpu = std::chrono::high_resolution_clock::now();
		    }

		    void endCpuTimer()
		    {
			    time_end_cpu = std::chrono::high_resolution_clock::now();

			    if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

			    std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
			    prev_elapsed_time_cpu_milliseconds =
				    static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

			    cpu_timer_started = false;
		    }

			void pauseTimer()
			{
				timer_pause = !timer_pause;

				if (timer_pause)
					time_pause_start_cpu = std::chrono::high_resolution_clock::now();
				else
				{
					time_pause_end_cpu = std::chrono::high_resolution_clock::now();

					std::chrono::duration<double, std::milli> duro = time_pause_end_cpu - time_pause_start_cpu;
					prev_elapsed_time_paused_milliseconds = static_cast<decltype(prev_elapsed_time_paused_milliseconds)>(duro.count());
				}

			}

		    void startGpuTimer()
		    {
			    if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
			    gpu_timer_started = true;

			    cudaEventRecord(event_start);
		    }

		    void endGpuTimer()
		    {
			    cudaEventRecord(event_end);
			    cudaEventSynchronize(event_end);

			    if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

			    cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
			    gpu_timer_started = false;
		    }

			void startGpuTimer2()
			{
				if (gpu_timer_started2) { throw std::runtime_error("GPU timer2 already started"); }
				gpu_timer_started2 = true;

				cudaEventRecord(event_start2);
			}

			void endGpuTimer2()
			{
				cudaEventRecord(event_end2);
				cudaEventSynchronize(event_end2);

				if (!gpu_timer_started2) { throw std::runtime_error("GPU timer2 not started"); }

				cudaEventElapsedTime(&prev_elapsed_time_gpu2_milliseconds, event_start2, event_end2);
				gpu_timer_started2 = false;
			}

		    float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
		    {
			    return prev_elapsed_time_cpu_milliseconds;
		    }

		    float getGpuElapsedTimeForPreviousOperation() //noexcept
		    {
			    return prev_elapsed_time_gpu_milliseconds;
		    }

			float getGpu2ElapsedTimeForPreviousOperation() //noexcept
			{
				return prev_elapsed_time_gpu2_milliseconds;
			}

		    // remove copy and move functions
		    PerformanceTimer(const PerformanceTimer&) = delete;
		    PerformanceTimer(PerformanceTimer&&) = delete;
		    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
		    PerformanceTimer& operator=(PerformanceTimer&&) = delete;

	    private:
		    cudaEvent_t event_start = nullptr;
		    cudaEvent_t event_end = nullptr;

			cudaEvent_t event_start2 = nullptr;
			cudaEvent_t event_end2 = nullptr;

			cudaEvent_t event_pause_start = nullptr;
			cudaEvent_t event_pause_end = nullptr;

		    using time_point_t = std::chrono::high_resolution_clock::time_point;
		    time_point_t time_start_cpu;
		    time_point_t time_end_cpu;

			time_point_t time_pause_start_cpu;
			time_point_t time_pause_end_cpu;

		    bool cpu_timer_started = false;
		    bool gpu_timer_started = false;
			bool gpu_timer_started2 = false;

			bool timer_pause = false;

		    float prev_elapsed_time_cpu_milliseconds = 0.f;
		    float prev_elapsed_time_gpu_milliseconds = 0.f;
			float prev_elapsed_time_gpu2_milliseconds = 0.f;

			float prev_elapsed_time_paused_milliseconds = 0.f;
	    };
    }
}
