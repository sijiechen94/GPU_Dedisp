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
#define WARP 32

__device__ __constant__ float c_DMs[1024];

//For f in GHz, DM in pc*cm^-3
#define OFFSET(f,DM) (int)rintf((4.149/f/f*DM)/dt)
#define RELATIVE_OFFSET(ch) (OFFSET((((ch)+blockIdx.x*TILE_WIDTH_F)*bandwidth+f_bot),DM) - OFFSET(f_bot,DM))


//The data is parallelised in the way that each block has every DM and covers few channels,
//while each thread in a single block has a particular DM.
//It seems that one block can hold only 1024 thread, so USE NO MORE THAN 1024 numDMs
//Each thread goes through all t and sum their channels to add to global accumulator
__global__ void timeshiftKernel(float* d_f_t, cufftComplex* d_dm_t, float f_bot, float bandwidth, 
				int tsize, int numchan, float dt, float* c_DMs){

	__device__ __shared__ float sharedInput[TILE_WIDTH_F][TILE_WIDTH_T];
	float DM = c_DMs[threadIdx.x];
	//The relative offset is quite small(<10000) so that every two of
	//them is forced to be stored in one 32-bits register to reduce the use
	//of registers.
	int off_0_1 = 65536* RELATIVE_OFFSET(0) + RELATIVE_OFFSET(1);
	int off_2_3 = 65536* RELATIVE_OFFSET(2) + RELATIVE_OFFSET(3);
	int off_4_5 = 65536* RELATIVE_OFFSET(4) + RELATIVE_OFFSET(5);
	int off_6_7 = 65536* RELATIVE_OFFSET(6) + RELATIVE_OFFSET(7);
	int off_8_9 = 65536* RELATIVE_OFFSET(8) + RELATIVE_OFFSET(9);
	int off_10_11 = 65536* RELATIVE_OFFSET(10) + RELATIVE_OFFSET(11);
	int off_12_13 = 65536* RELATIVE_OFFSET(12) + RELATIVE_OFFSET(13);
	int off_14_15 = 65536* RELATIVE_OFFSET(14) + RELATIVE_OFFSET(15);
	
	for ( int t=tsize; t>TILE_WIDTH_T; t-=TILE_WIDTH_T ){
		
		__syncthreads();
		///Note that only the first 32 threads (first warp) is used to load,
		///so make sure your data RUN AT LEAST 32 DMs
		if(threadIdx.x<WARP)
			for ( int i=0; i<TILE_WIDTH_T*TILE_WIDTH_F/WARP; i++ )	sharedInput[0][i*WARP+threadIdx.x] = 
				d_f_t[(blockIdx.x*TILE_WIDTH_F + (i*WARP+threadIdx.x)/TILE_WIDTH_T)*tsize 
 					+ ((i*WARP+threadIdx.x)%TILE_WIDTH_T ) + (t-TILE_WIDTH_T) ];
		__syncthreads();
		
		for ( int tj=0; tj<TILE_WIDTH_T; tj++ ) {

			//This loop has been manually broken down to make sure that
			//all the offsets are stored in registers, for once you use a
			//loop, the compiler won't be able to determine the addresses
			//at compile time so that the offsets must be settled in global
			//memory and the load will be much slower. These code might be
			//extremely ugly and hard to maintain but it's very important.
			
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_0_1/65536)
				  ,sharedInput[0][tj+0*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_0_1%65536)
				  ,sharedInput[0][tj+1*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_2_3/65536)
				  ,sharedInput[0][tj+2*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_2_3%65536)
				  ,sharedInput[0][tj+3*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_4_5/65536)
				  ,sharedInput[0][tj+4*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_4_5%65536)
				  ,sharedInput[0][tj+5*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_6_7/65536)
				  ,sharedInput[0][tj+6*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_6_7%65536)
				  ,sharedInput[0][tj+7*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_8_9/65536)
				  ,sharedInput[0][tj+8*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_8_9%65536)
				  ,sharedInput[0][tj+9*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_10_11/65536)
				  ,sharedInput[0][tj+10*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_10_11%65536)
				  ,sharedInput[0][tj+11*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_12_13/65536)
				  ,sharedInput[0][tj+12*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_12_13%65536)
				  ,sharedInput[0][tj+13*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_14_15/65536)
				  ,sharedInput[0][tj+14*TILE_WIDTH_F]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_14_15%65536)
				  ,sharedInput[0][tj+15*TILE_WIDTH_F]);

			///Note that atomicAdd may hinder performance, especially on old cards
			///However, race conditions only happens between different BLOCKS,
			///so if you are running only one block at any time, use normal add.
			///And even you're running not too many (4 or 5) blocks, the chance
			///of race conditions happening is quite negligible.
			///PS: When you are feeling lucky, use normal add:

			/***  
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_0_1/65536)).x+=sharedInput[0][tj+0*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_0_1%65536)).x+=sharedInput[0][tj+1*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_2_3/65536)).x+=sharedInput[0][tj+2*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_2_3%65536)).x+=sharedInput[0][tj+3*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_4_5/65536)).x+=sharedInput[0][tj+4*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_4_5%65536)).x+=sharedInput[0][tj+5*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_6_7/65536)).x+=sharedInput[0][tj+6*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_6_7%65536)).x+=sharedInput[0][tj+7*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_8_9/65536)).x+=sharedInput[0][tj+8*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_8_9%65536)).x+=sharedInput[0][tj+9*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_10_11/65536)).x+=sharedInput[0][tj+10*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_10_11%65536)).x+=sharedInput[0][tj+11*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_12_13/65536)).x+=sharedInput[0][tj+12*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_12_13%65536)).x+=sharedInput[0][tj+13*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_14_15/65536)).x+=sharedInput[0][tj+14*TILE_WIDTH_F];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_14_15%65536)).x+=sharedInput[0][tj+15*TILE_WIDTH_F];
			***/

		}
	};
}

void dedispersion(float* f_t, int numchan, int tsize,
		  float f_ctr, float bandwidth, float dt, float* DMs, int numDMs){

	//Assume f_t is a 2D Array Input[f][t]

	float sizeofDMArray = numDMs*sizeof(float);
	cudaCheckErrors(cudaMemcpyToSymbol(c_DMs,DMs,sizeofDMArray,0,cudaMemcpyHostToDevice));

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
