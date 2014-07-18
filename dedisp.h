#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"
#include "cufft.h"
#include "curand_kernel.h"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


__global__ void setComplexZero(cufftComplex* d_dest, int arraysize);

__global__ void timeshiftKernel(float* d_f_t, cufftComplex* d_dm_t, float f_bot, float bandwidth, 
				int tsize, int numchan, float dt, float* c_DMs);


void dedispersion(float* f_t, int numchan, int tsize,
		  float f_ctr, float bandwidth, float dt, float* DMs, int numDMs, float* output_dm_t);
