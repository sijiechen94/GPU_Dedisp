#include "cuda.h"
#include "stdio.h"
#include "limits.h"
#include "stdlib.h"
#include "cufft.h"

#define CUDAERROR(err)\
	 if(err!=cudaSuccess)\
	 {printf("%s in %s at line %d\n",\
	 cudaGetErrorString(err),__FILE__,__LINE__);\
	 exit(EXIT_FAILURE);}

//Tile properties, size of each tile is 48KB,
//if your shared memory is smaller, make these smaller
#define TILE_WIDTH_F 16
#define TILE_WIDTH_T 768

#define OFFSET(f,DM) (int)rintf(0.00001/f/f*DM)

//The data is parallelised in the way that each block has every DM and covers few channels,
//while each thread in a single block has a particular DM.
//Each thread goes through all t and sum their channels to add to global accumulator
__global__ void timeshiftKernel(float* d_f_t, cufftComplex* d_dm_t, float f_bot, float bandwidth, 
				int tsize, int numchan, float coefA, float coefB){

	//Assume DM is a linear function of threadIdx.x
	float DM = coefA*threadIdx.x + coefB;

	//define tile properties to be loaded into shared memory each time
	float f_tile_low = TILE_WIDTH_F * blockIdx.x * bandwidth + f_bot;
	float f_tile_high = (TILE_WIDTH_F * blockIdx.x + TILE_WIDTH_F - 1) * bandwidth + f_bot;
	int t_tile_step = TILE_WIDTH_T - (OFFSET(f_tile_low,(coefA*blockDim.x-coefA+coefB))-OFFSET(f_tile_high,coefB));
		//t_tile_step = TILE_WIDTH_T - (Largest_Offset - Smallest_Offset)
	int t_tile_begin = OFFSET(f_tile_high,DM)-t_tile_step;
	__device__ __shared__ float sharedInput[TILE_WIDTH_F][TILE_WIDTH_T];
	
	for ( int t=0; t<tsize; t++ ){
		
		//Every time the data to be accessed is out of the shared memory,
		//reload some data from global memory. It is assumed numDMs is a multiple of 32,
		//so that thread scheduling is made most efficient. 
		if(t%t_tile_step==0){
			t_tile_begin+=t_tile_step;
			/****Caution****/
			//12288%blockDim.x=0, so that blockDim.x can only have values like
			//32, 64, 128... or 96, 192...., otherwise sharedMem can't be fully loaded
			for ( int i=0; i<TILE_WIDTH_F*TILE_WIDTH_T/blockDim.x; i++ )
				//This loading is coalesced
				sharedInput[0][i*blockDim.x+threadIdx.x] = d_f_t[
					(blockIdx.x*TILE_WIDTH_F + (i*blockDim.x+threadIdx.x)/TILE_WIDTH_T)*tsize 
 					+ (t_tile_begin + (i*blockDim.x+threadIdx.x)%TILE_WIDTH_T ) ];
			__syncthreads();
		}

		float localSum = 0;
		(*(d_dm_t+tsize*threadIdx.x+t)).x=0;
		(*(d_dm_t+tsize*threadIdx.x+t)).y=0;
		for ( int ch=0; ch<TILE_WIDTH_F; ch++)
			localSum += sharedInput[ch][t+OFFSET((f_tile_low+ch*bandwidth),DM)];
		atomicAdd(&((*(d_dm_t+tsize*threadIdx.x+t)).x),localSum);
		__syncthreads();
	};
}

void dedispersion(float* f_t, int numchan, int tsize,
		  float f_ctr, float bandwidth){

	//Assume f_t is a 2D Array Input[f][t]

	///Make Time Shift Plan
	unsigned numDMs=128;
	unsigned coefA=1;
	unsigned coefB=0;
	///Temporarily, no subbands involved

	float* d_f_t;
	cufftComplex* d_dm_t;
	unsigned input_size = numchan*tsize*sizeof(float);
	unsigned output_size = numDMs*tsize*sizeof(cufftComplex);

	cudaError_t err = cudaMalloc((void**)&d_f_t,input_size);
	CUDAERROR(err)
	err = cudaMalloc((void**)&d_dm_t,output_size);
	CUDAERROR(err)
	cudaMemcpy(d_f_t,f_t,input_size,cudaMemcpyHostToDevice);

	dim3 dimBlock(numDMs,1,1);
	dim3 dimGrid(numchan/TILE_WIDTH_F,1,1);
	timeshiftKernel<<<dimGrid,dimBlock>>>(d_f_t,d_dm_t,f_ctr-numchan*bandwidth/2,bandwidth,tsize,numchan,coefA,coefB);

	cudaFree(d_f_t);
	float* h_dm_t;
	h_dm_t = (float*)malloc(output_size);
	cudaMemcpy(h_dm_t,d_dm_t,output_size,cudaMemcpyDeviceToHost);
	//Write Y(DM,t) to files
	
	cufftHandle plan;
	cufftPlanMany(&plan,1,&tsize,NULL,0,0,NULL,0,0,CUFFT_C2C,numDMs);
	
	cufftExecC2C(plan,d_dm_t,d_dm_t,CUFFT_FORWARD);
	cufftDestroy(plan);
	cudaMemcpy(h_dm_t,d_dm_t,output_size,cudaMemcpyDeviceToHost);
	//Write Z(DM,f) to files

	cudaFree(d_dm_t);
}
