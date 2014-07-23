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

\\\\\\

It seems that atomicAdd do work faster than normal add:
[chensijie@n04 ~]$ vi dedisp.cu
[chensijie@n04 ~]$ nvcc -arch=sm_20 dedisp.cu -lcufft
[chensijie@n04 ~]$ time ./a.out
Set to zero...

Start de-dispersion!
Input wrote to f_t.txt, in 1024*196608 float
Time series wrote to DM_t.txt, in 256*196608 float
Frequency series wrote to DM_f.txt, in 256*196608 float

real	2m17.665s
user	1m26.767s
sys	0m48.475s
[chensijie@n04 ~]$ vi dedisp.cu
[chensijie@n04 ~]$ nvcc -arch=sm_20 dedisp.cu -lcufft
[chensijie@n04 ~]$ time ./a.out
Set to zero...

Start de-dispersion!
Input wrote to f_t.txt, in 1024*196608 float
Time series wrote to DM_t.txt, in 256*196608 float
Frequency series wrote to DM_f.txt, in 256*196608 float

real	1m54.308s
user	1m11.761s
sys	0m38.543s
