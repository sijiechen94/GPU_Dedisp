#include <limits.h>
#include <ctype.h>
#include "presto.h"
#include "readfile_cmd.h"
#include "mask.h"
#include "backend_common.h"
#include "psrfits.h"

extern int is_PSRFITS(char *filename);

void print_rawbincand(rawbincand cand);

/* A few global variables */

long N;
double dt, nph;

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif

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

   // Use Backend reading stuff
   struct spectra_info s;

   spectra_info_set_defaults(&s);
   s.datatype=PSRFITS;
   s.filenames = argv+1;
   s.num_files = argc-1;
   read_PSRFITS_files(&s);
   printf("Info loaded.\n");
   //A lot of things to do here...
   exit(0);
}
