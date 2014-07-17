#include <limits.h>
#include <ctype.h>
#include "presto.h"
#include "readfile_cmd.h"
#include "mask.h"
#include "backend.h"
#include "psrfits.h"
#include "dedisp.h"

extern int is_PSRFITS(char *filename);

#define POWER(r,i) pow(((r)*(r)+(i)*(i)),0.5)

int main(int argc, char **argv)
{
    if (argc == 1) exit(0);
    
    /* Check data type from the file name */
    
    if (is_PSRFITS(argv[1])) {
        fprintf(stdout,"Assuming the data is in PSRFITS format.\n\n");
    }else {
        fprintf(stdout,"Your data don't seem like PSRFITS format.\n\n");
        exit(0);
    }
    
    struct spectra_info s;
    spectra_info_set_defaults(&s);
    s.datatype=PSRFITS;
    s.filenames = argv+1;
    s.num_files = argc-1;
    read_PSRFITS_files(&s);
    printf("Info loaded.\n");
    
    float *fdata;
    fdata = (float*)malloc(s.N*s.num_channels*sizeof(float));
    
    int ii, numvals, gotblock, pad ;
    for (ii = 0; ii < s.num_files; ii++) {
        numvals = s.spectra_per_subint * s.num_channels;
        pad=0;
        for (int jj = 0; jj < s.num_subint[ii]; jj++) {
            gotblock = get_PSRFITS_rawblock(fdata + jj * numvals, &s, &pad);
            if (gotblock==0) break;
        }
    }
    
    //A lot of things to do here...
    
    FILE* fp=fopen("/nfshome/chensijie/t.txt","w");
	FILE* fp2=fopen("/nfshome/chensijie/f.txt","w");
    
    
	int i=0, numDMs=32;
    float *DMs=gen_fvect(numDMs);
	for (i=0;i<32;i++) *(DMs+i)=i;
    float *t_output,*f_output;
	t_output = (float*)malloc(2*s.N*numDMs*sizeof(float));
	f_output = (float*)malloc(2*s.N*numDMs*sizeof(float));
    
    
    dedispersion(fdata, s.num_channels, s.N,
                 s.lo_freq+(s.num_channels-1)*s.df, s.df, s.dt, DMs, numDMs, t_output, f_output);
    
    //Print output
	for (i=0; i<s.num_channels*s.N; i++)
        fprintf(fp,"%.0f%c",*(fdata+i),(i+1)%s.N?'\t':'\n');
	for (i=0; i<numDMs*s.N; i++)
        fprintf(fp,"%.0f%c",t_output[2*i],(i+1)%s.N?'\t':'\n');
	for (i=0; i<numDMs*s.N; i++)
        fprintf(fp2,"%.0f%c",POWER(f_output[2*i],f_output[2*i+1]),(i+1)%s.N?'\t':'\n');
	fclose(fp);
	fclose(fp2);
    
    exit(0);
}
