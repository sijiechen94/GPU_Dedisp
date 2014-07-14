#include "dedisp.h"
#include "math.h"

//Tile properties, size of each tile is 48KB,
//if your shared memory is smaller, make these smaller
#define TILE_WIDTH_F 16
#define TILE_WIDTH_T 768
#define WARP 32

__device__ __constant__ float c_DMs[1024];

//For f in GHz, DM in pc*cm^-3
#define OFFSET(f,DM) roundf((4.149/(f)/(f)*(DM))/(dt))
#define RELATIVE_OFFSET(ch) roundf((OFFSET((f_ctr-bandwidth*((numchan-1)/2.0-(ch)-0*TILE_WIDTH_F)),DM) - OFFSET(f_ctr+(numchan-1)/2.0*bandwidth,DM)))


//The data is parallelised in the way that each block has every DM and covers few channels,
//while each thread in a single block has a particular DM.
//It seems that one block can hold only 1024 thread, so USE NO MORE THAN 1024 numDMs
//Each thread goes through all t and sum their channels to add to global accumulator


__global__ void setComplexZero(cufftComplex* d_dest, int arraysize){
	for(int i=0; i<arraysize; i++)	*(d_dest+i+arraysize*threadIdx.x) = (cufftComplex){0,0};
}

__global__ void timeshiftKernel(float* d_f_t, cufftComplex* d_dm_t, float f_ctr, float bandwidth, 
				int tsize, int numchan, float dt){

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
	for ( int t=tsize; t>=TILE_WIDTH_T; t-=TILE_WIDTH_T ){
		
		__syncthreads();
		///Note that only the first 32 threads (first warp) is used to load,
		///so make sure your data RUN AT LEAST 32 DMs
		if(threadIdx.x<WARP)
			for ( int i=0; i<TILE_WIDTH_T*TILE_WIDTH_F/WARP; i++ )	sharedInput[0][i*WARP+threadIdx.x] = 
				*(d_f_t+(blockIdx.x*TILE_WIDTH_F + (i*WARP+threadIdx.x)/TILE_WIDTH_T)*tsize 
 					+ ((i*WARP+threadIdx.x)%TILE_WIDTH_T ) + (t-TILE_WIDTH_T));
		__syncthreads();
		
		for ( int tj=0; tj<TILE_WIDTH_T; tj++ ) {

			//This loop has been manually broken down to make sure that
			//all the offsets are stored in registers, for once you use a
			//loop, the compiler won't be able to determine the addresses
			//at compile time so that the offsets must be settled in global
			//memory and the load will be much slower. These code might be
			//extremely ugly and hard to maintain but it's very important.
			if(t-TILE_WIDTH_T+tj-off_0_1/65536>0){
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_0_1/65536) ,sharedInput[0][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_0_1%65536) ,sharedInput[1][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_2_3/65536) ,sharedInput[2][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_2_3%65536) ,sharedInput[3][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_4_5/65536) ,sharedInput[4][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_4_5%65536) ,sharedInput[5][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_6_7/65536) ,sharedInput[6][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_6_7%65536) ,sharedInput[7][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_8_9/65536) ,sharedInput[8][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_8_9%65536) ,sharedInput[9][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_10_11/65536) ,sharedInput[10][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_10_11%65536) ,sharedInput[11][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_12_13/65536) ,sharedInput[12][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_12_13%65536) ,sharedInput[13][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_14_15/65536) ,sharedInput[14][tj]);
			atomicAdd((float*)(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_14_15%65536) ,sharedInput[15][tj]);

			//Note that atomicAdd may hinder performance, especially on old cards
			//However, race conditions only happens between different BLOCKS,
			//so if you are running only one block at any time, use normal add.
			//And even you're running not too many (4 or 5) blocks, the chance
			//of race conditions happening is quite negligible.
			//PS: When you are feeling lucky, use normal add:

			
			/*(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_0_1/65536)).x+=sharedInput[0][tj+0*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_0_1%65536)).x+=sharedInput[0][tj+1*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_2_3/65536)).x+=sharedInput[0][tj+2*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_2_3%65536)).x+=sharedInput[0][tj+3*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_4_5/65536)).x+=sharedInput[0][tj+4*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_4_5%65536)).x+=sharedInput[0][tj+5*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_6_7/65536)).x+=sharedInput[0][tj+6*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_6_7%65536)).x+=sharedInput[0][tj+7*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_8_9/65536)).x+=sharedInput[0][tj+8*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_8_9%65536)).x+=sharedInput[0][tj+9*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_10_11/65536)).x+=sharedInput[0][tj+10*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_10_11%65536)).x+=sharedInput[0][tj+11*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_12_13/65536)).x+=sharedInput[0][tj+12*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_12_13%65536)).x+=sharedInput[0][tj+13*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_14_15/65536)).x+=sharedInput[0][tj+14*TILE_WIDTH_T];
			(*(d_dm_t+tsize*threadIdx.x+t-TILE_WIDTH_T+tj-off_14_15%65536)).x+=sharedInput[0][tj+15*TILE_WIDTH_T];*/
			
			}
		}
	};
}

void dedispersion(float* f_t, int numchan, int tsize,
		  float f_ctr, float bandwidth, float dt, float* DMs, int numDMs, float* output_dm_t, float* output_f_t){

	//Assume f_t is a 2D Array Input[f][t]

	//Write to __constant__ memory
	int sizeofDMArray = numDMs*sizeof(float);
	cudaMemcpyToSymbol(c_DMs,DMs,sizeofDMArray,0,cudaMemcpyHostToDevice);

	int input_size = numchan*tsize*sizeof(float);
	int output_size = numDMs*tsize*sizeof(cufftComplex);

	//Allocate device memory for input and output
	float* d_f_t;				//input
	cufftComplex* d_dm_t;			//output
	cudaMalloc((void**)&d_f_t,input_size);
	cudaMalloc((void**)&d_dm_t,output_size);
	//Copy host memory to device memory
	cudaMemcpy(d_f_t,f_t,input_size,cudaMemcpyHostToDevice);

	//Launch a small kernel to initialize output array
	dim3 dimBlock(numDMs,1,1);
	dim3 dimGrid(1,1,1);
	setComplexZero<<<1,32>>>(d_dm_t,tsize);

	//Launch main kernel to do time shift
	dim3 dimGrid2(numchan/TILE_WIDTH_F,1,1);
	timeshiftKernel<<<dimGrid2,dimBlock>>>(d_f_t,d_dm_t,f_ctr,bandwidth,tsize,numchan,dt);

	//Copy output from device memory to host memory
	cudaFree(d_f_t);
	cudaMemcpy((void*)output_dm_t,(void*)d_dm_t,output_size,cudaMemcpyDeviceToHost);
	
	//Do FFT
	cufftHandle plan;
	int n[1]={tsize};
	cufftPlanMany(&plan,1,n,n,1,tsize,n,1,tsize,CUFFT_C2C,numDMs);
	
	cufftExecC2C(plan,d_dm_t,d_dm_t,CUFFT_FORWARD);
	cufftDestroy(plan);
	cudaMemcpy(output_f_t,d_dm_t,output_size,cudaMemcpyDeviceToHost);
	//Write Z(DM,f) to files
	
	//DEBUG
	cudaFree(d_dm_t);
	printf("End Of Kernel!\n");
}

#define POWER(r,i) pow(((r)*(r)+(i)*(i)),0.5)

int main(){
	FILE* fp=fopen("/nfshome/chensijie/t.txt","w");
	FILE* fp2=fopen("/nfshome/chensijie/f.txt","w");

	//Initialize observation
	int numchan = TILE_WIDTH_F;
	int numsignal = 8;
	int tsize = TILE_WIDTH_T*numsignal;
	float f_ctr = 17.5;
	float bandwidth = 1;
	int numDMs = 32;
	float dt = 0.001;

	int k;
	//Initialize DM array
	float *DMs;
	DMs = (float*)malloc(32*sizeof(float));
	int i=0;
	for (i=0;i<32;i++) *(DMs+i)=i;

	//Initialize input array(Fake data)
	float* f_t=NULL;
	f_t = (float*)malloc(sizeof(float)*TILE_WIDTH_F*tsize);
	for ( k=0;k<TILE_WIDTH_F*tsize;k++)
		*(f_t+k)=0.0f;
	printf("Set to zero...\n");
	for ( k=0;k<16;k++){
		float f=f_ctr-((numchan-1)/2.0-k)*bandwidth;
		for ( i=0 ; i<numsignal; i++)
		*(f_t+k*tsize+200+i*768+(int)round(4.149/f/f*4/0.001))=1000;
	}

	float *t_output,*f_output;
	t_output = (float*)malloc(2*tsize*numDMs*sizeof(float));
	f_output = (float*)malloc(2*tsize*numDMs*sizeof(float));
	

	printf("\nStart de-dispersion!\n");	
	dedispersion(f_t, numchan, tsize, f_ctr, bandwidth,dt, DMs, numDMs , t_output, f_output);

	//Print output
	for (i=0; i<numchan*tsize; i++)
	fprintf(fp,"%.0f%c",*(f_t+i),(i+1)%tsize?'\t':'\n');
	//for (i=0; i<numDMs*tsize; i++)
	//fprintf(fp,"%.0f%c",t_output[2*i],(i+1)%tsize?'\t':'\n');
	for (i=0; i<numDMs*tsize; i++)
	fprintf(fp2,"%.0f%c",POWER(f_output[2*i],f_output[2*i+1]),(i+1)%tsize?'\t':'\n');
	fclose(fp);
	fclose(fp2);
}
