#This is now for test only!!!

Use this command to compile the source:

nvcc -arch=sm_20 dedisp.cu -L$CUDA_TOOLKIT_ROOT_DIR/lib64 -lcufft
              ||
              ||Here Computing Capability 2.0 is the least requirement for the use of 
                    atomicAdd() for float,
		if your device support higher architecture, set it to a higher level,
		if 2.0 is not supported, change the source file to use normal add

Other usage requirements are listed in source file.
