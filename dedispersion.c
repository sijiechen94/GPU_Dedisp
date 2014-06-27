#include "cuda.h"
#include "stdio.h"
#include "limits.h"
#include "stdlib.h"
//Each thread has some particular DM
__global__ void timeshiftKernel(){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	//DM = SOME_FUNCTION(index);
	for ( int t=0; t<tsize; t++ ){
		//Transfer some data to __shared__ memory and __synchronize__
		for ( int i=0; i<numchan; i++){
			//float f = SOME_OTHER_FUNCTION(i,f_ctr,bandwidth);
			//int offset = SOME_FUNCTION(f,DM);
			//SOME_TEMP_MEMSPACE[index][t] += SOME_OTHER_TEMP_MEMSPACE[i][t+offset];
		}
		//Transfer some data from __shared__ memory and __synchronize__
	}
	//Make sure the Y(DM,t) is ready
}

void dedispersion(float* f_t, int numchan, int tsize,
		  float f_ctr, float bandwidth){

	///Make Time Shift Plan

	float* d_f_t;
	float* d_dm_t;
	unsigned input_size = numchan*tsize*sizeof(float);
	unsigned output_size = dms*tsize*sizeof(float);

	cudaMalloc((void**)&d_f_t,input_size);
	cudaMemcpy(d_f_t,f_t,input_size,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_dm_t,output_size);

	///Time Shift Kernel Invocation

	cudaFree(d_f_t);
	float* h_dm_t;
	h_dm_t = malloc(output_size);
	cudaMemcpy(h_dm_t,d_dm_t,output_size,cudaMemcpyDeviceToHost);
	
	///Make cuFFT Plan

	///Execute cuFFT
	///Meanwhile write Y(DM,t) to files

	cudaFree(d_dm_t);
	free(dm_t)

}
