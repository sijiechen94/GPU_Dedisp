#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"
#include "cufft.h"

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

//Tile properties, size of each tile is 48KB,
//if your shared memory is smaller, make these smaller
#define TILE_WIDTH_F 16
#define TILE_WIDTH_T 768

__device__ __constant__ float c_DMs[1024];

//For f in GHz, DM in pc*cm^-3
#define OFFSET(f,DM) (int)rintf((4.149/f/f*DM)/dt)

//The data is parallelised in the way that each block has every DM and covers few channels,
//while each thread in a single block has a particular DM.
//Each thread goes through all t and sum their channels to add to global accumulator
__global__ void timeshiftKernel(float* d_f_t, cufftComplex* d_dm_t, float f_bot, float bandwidth, 
				int tsize, int numchan, float dt, float* c_DMs){

	__device__ __shared__ float sharedInput[TILE_WIDTH_F][TILE_WIDTH_T];
	
	for ( int t=tsize; t>TILE_WIDTH_T; t-=TILE_WIDTH_T ){
		
		__syncthreads();
		///Note that only the first 32 threads (first warp) is used to load,
		///so make sure your data RUN AT LEAST 32 DMs
		if(threadIdx.x<32)
			for ( int i=0; i<384; i++ )	sharedInput[0][i*32+threadIdx.x] = 
				d_f_t[(blockIdx.x*TILE_WIDTH_F + (i*32+threadIdx.x)/TILE_WIDTH_T)*tsize 
 					+ ((i*32+threadIdx.x)%TILE_WIDTH_T ) + (t-TILE_WIDTH_T) ];
		__syncthreads();
		
		for ( int tj=0; tj<TILE_WIDTH_T; tj++ ) 
			for ( int ch=0; ch<TILE_WIDTH_F; ch++){
				///Note that atomicAdd may hinder performance, especially on old cards
				atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-OFFSET((ch+blockIdx.x*TILE_WIDTH_F),(c_DMs[threadIdx.x])))
					  ,sharedInput[0][tj+ch*TILE_WIDTH_F]);
				///However, race conditions only happens between different BLOCKS,
				///so if you are running only one block at any time, use normal add.
				///And even you're running not too many (4 or 5) blocks, the chance
				///of race conditions happening is quite negligible.
				///PS: When you are feeling lucky, use normal add:
				/***    (*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-OFFSET((ch+blockIdx.x*TILE_WIDTH_F),(c_DMs[threadIdx.x])))).x
					+=sharedInput[0][tj+ch*TILE_WIDTH_F];    ***/
			}
	};
}

void dedispersion(float* f_t, int numchan, int tsize,
		  float f_ctr, float bandwidth, float dt, float* DMs, int numDMs){

	//Assume f_t is a 2D Array Input[f][t]

	///Make Time Shift Plan
	float sizeofDMArray = numDMs*sizeof(float);
	cudaCheckErrors(cudaMemcpyToSymbol(c_DMs,DMs,sizeofDMArray,0,cudaMemcpyHostToDevice));
	///Temporarily, no subbands involved

	float* d_f_t;
	cufftComplex* d_dm_t;
	unsigned input_size = numchan*tsize*sizeof(float);
	unsigned output_size = numDMs*tsize*sizeof(cufftComplex);

	cudaCheckErrors(cudaMalloc((void**)&d_f_t,input_size));
	cudaCheckErrors(cudaMalloc((void**)&d_dm_t,input_size));
	cudaCheckErrors(cudaMemcpy(d_f_t,f_t,input_size,cudaMemcpyHostToDevice));

	dim3 dimBlock(numDMs,1,1);
	dim3 dimGrid(numchan/TILE_WIDTH_F,1,1);
	timeshiftKernel<<<dimGrid,dimBlock>>>(d_f_t,d_dm_t,f_ctr-numchan*bandwidth/2,bandwidth,tsize,numchan,dt,c_DMs);

	cudaFree(d_f_t);
	float* h_dm_t;
	h_dm_t = (float*)malloc(output_size);
	cudaCheckErrors(cudaMemcpy(h_dm_t,d_dm_t,output_size,cudaMemcpyDeviceToHost));
	//Write Y(DM,t) to files
	
	cufftHandle plan;
	cudaCheckErrors(cufftPlanMany(&plan,1,&tsize,NULL,0,0,NULL,0,0,CUFFT_C2C,numDMs));
	
	cudaCheckErrors(cufftExecC2C(plan,d_dm_t,d_dm_t,CUFFT_FORWARD));
	cufftDestroy(plan);
	cudaCheckErrors(cudaMemcpy(h_dm_t,d_dm_t,output_size,cudaMemcpyDeviceToHost));
	//Write Z(DM,f) to files

	cudaFree(d_dm_t);
}
