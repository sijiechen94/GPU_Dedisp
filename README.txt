This program is aimed to do de-dispersion on PSRFITS raw data fast with the aid of CUDA and NVIDIA GPUs.

This work is done by a undergraduate student during a one-month summer internship who is facing parallel computation for the first time. The problem it can handle is rather limited. It take a input as a 2D floating array, a array of DMs to try, major spectra information, and addresses to return the output. Many arguments is limit to have certain values for the program to work at best condition. These requirements are listed in source file.

Use this command to compile the source:

nvcc -arch=sm_20 dedisp.cu -lcufft
              ||
              ||Here Computing Capability 2.0 is the least requirement for the use of 
                    atomicAdd() for float,
		if your device support higher architecture, set it to a higher level,
		if 2.0 is not supported, change the source file to use normal add

If you failed to link to cufft, check if your $PATH has the correct lib path.

Other usage requirements are listed in source file.

If you have any ideas or problems, contact me:
Sijie Chen <kevinsouldew@gmail.com>
