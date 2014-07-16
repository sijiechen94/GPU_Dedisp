#include <sys/types.h>
#include <pwd.h>
#include "presto.h"
#include "mask.h"
#include "psrfits.h"
#include "fitsio.h"

#define DEBUG_OUT 1

static unsigned char *cdatabuffer;
static float *fdatabuffer, *offsets, *scales, *weights;
static int cur_file = 0, cur_subint = 1, numbuffered = 0;
static long long cur_spec = 0, new_spec = 0;

extern double slaCldj(int iy, int im, int id, int *j);
extern short transpose_bytes(unsigned char *a, int nx, int ny, unsigned char *move,
                             int move_size);
extern void add_padding(float *fdata, float *padding, int numchan, int numtopad);

void get_PSRFITS_subint(float *fdata, unsigned char *cdata, 
                        struct spectra_info *s);



int is_PSRFITS(char *filename)
// Return 1 if the file described by filename is a PSRFITS file
// Return 0 otherwise.                                         
{
    fitsfile *fptr;
    int status=0;
    char ctmp[80], comment[120];
    
    // Read the primary HDU
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) return 0;
    
    // Make the easy check first
    fits_read_key(fptr, TSTRING, "FITSTYPE", ctmp, comment, &status);
    if (status || strcmp(ctmp, "PSRFITS")) return 0;

    // See if the data are search-mode
    fits_read_key(fptr, TSTRING, "OBS_MODE", ctmp, comment, &status);
    if (status || (strcmp(ctmp, "SEARCH") && 
                   strcmp(ctmp, "SRCH"))) return 0;

    fits_close_file(fptr, &status);
    return 1;  // it is search-mode  PSRFITS
}

#define get_hdr_string(name, param) {                                   \
        fits_read_key(s->fitsfiles[ii], TSTRING, (name), ctmp, comment, &status); \
        if (status) {\
            printf("Error %d reading key %s\n", status, name); \
            if (ii==0) param[0]='\0'; \
            if (status==KEY_NO_EXIST) status=0;                      \
        } else {                                                     \
            if (ii==0) strncpy((param), ctmp, 40);                          \
            else if (strcmp((param), ctmp)!=0)                              \
                printf("Warning!:  %s values don't match for files 0 and %d!\n", \
                       (name), ii);                                         \
        }                                                               \
    }

#define get_hdr_int(name, param) {                                      \
        fits_read_key(s->fitsfiles[ii], TINT, (name), &itmp, comment, &status); \
        if (status) {\
            printf("Error %d reading key %s\n", status, name); \
            if (ii==0) param=0; \
            if (status==KEY_NO_EXIST) status=0;\
        } else {                                                          \
            if (ii==0) param = itmp;                                        \
            else if (param != itmp)                                         \
                printf("Warning!:  %s values don't match for files 0 and %d!\n", \
                       (name), ii);                                         \
        }                                                               \
    }

#define get_hdr_double(name, param) {                                   \
        fits_read_key(s->fitsfiles[ii], TDOUBLE, (name), &dtmp, comment, &status); \
        if (status) {\
            printf("Error %d reading key %s\n", status, name); \
            if (ii==0.0) param=0.0; \
            if (status==KEY_NO_EXIST) status=0;\
        } else {                                                          \
            if (ii==0) param = dtmp;                                        \
            else if (param != dtmp)                                         \
                printf("Warning!:  %s values don't match for files 0 and %d!\n", \
                       (name), ii);                                         \
        }                                                               \
    }



void read_PSRFITS_files(struct spectra_info *s)
// Read and convert PSRFITS information from a group of files 
// and place the resulting info into a spectra_info structure.
{
    int IMJD, SMJD, itmp, ii, status = 0;
    double OFFS, dtmp;
    long double MJDf;
    char ctmp[80], comment[120];
    
    s->datatype = PSRFITS;
    s->fitsfiles = (fitsfile **)malloc(sizeof(fitsfile *) * s->num_files);
    s->start_subint = gen_ivect(s->num_files);
    s->num_subint = gen_ivect(s->num_files);
    s->start_spec = (long long *)malloc(sizeof(long long) * s->num_files);
    s->num_spec = (long long *)malloc(sizeof(long long) * s->num_files);
    s->num_pad = (long long *)malloc(sizeof(long long) * s->num_files);
    s->start_MJD = (long double *)malloc(sizeof(long double) * s->num_files);
    s->N = 0;
    s->num_beams = 1;
    //s->get_rawblock = &get_PSRFITS_rawblock;
    //s->offset_to_spectra = &offset_to_PSRFITS_spectra;

    // Step through the other files
    for (ii = 0 ; ii < s->num_files ; ii++) {

        // Open the PSRFITS file
        fits_open_file(&(s->fitsfiles[ii]), s->filenames[ii], READONLY, &status);

        // Is the data in search mode?
        fits_read_key(s->fitsfiles[ii], TSTRING, "OBS_MODE", ctmp, comment, &status);
        if (strcmp(ctmp, "SEARCH")) {
            fprintf(stderr, 
                    "\nError!  File '%s' does not contain SEARCH-mode data!\n", 
                    s->filenames[ii]);
            exit(1);
        }

        // Now get the stuff we need from the primary HDU header
        fits_read_key(s->fitsfiles[ii], TSTRING, "TELESCOP", ctmp, comment, &status); \
        if (status) {
            printf("Error %d reading key %s\n", status, "TELESCOP");
            if (ii==0) s->telescope[0]='\0';
            if (status==KEY_NO_EXIST) status=0;
        } else {
            if (ii==0) strncpy(s->telescope, ctmp, 40);
            else if (strcmp(s->telescope, ctmp)!=0)
                printf("Warning!:  %s values don't match for files 0 and %d!\n",
                       "TELESCOP", ii);
        }

        get_hdr_string("OBSERVER", s->observer);
        get_hdr_string("SRC_NAME", s->source);
        get_hdr_string("FRONTEND", s->frontend);
        get_hdr_string("BACKEND", s->backend);
        get_hdr_string("PROJID", s->project_id);
        get_hdr_string("DATE-OBS", s->date_obs);
        get_hdr_string("FD_POLN", s->poln_type);
        get_hdr_string("RA", s->ra_str);
        get_hdr_string("DEC", s->dec_str);
        get_hdr_double("OBSFREQ", s->fctr);
        get_hdr_int("OBSNCHAN", s->orig_num_chan);
        get_hdr_double("OBSBW", s->orig_df);
        //get_hdr_double("CHAN_DM", s->chan_dm);
        get_hdr_double("BMIN", s->beam_FWHM);

        /* This is likely not in earlier versions of PSRFITS so */
        /* treat it a bit differently                           */
        fits_read_key(s->fitsfiles[ii], TDOUBLE, "CHAN_DM", 
                      &(s->chan_dm), comment, &status);
        if (status==KEY_NO_EXIST) {
            status = 0;
            s->chan_dm = 0.0;
        }

        // Don't use the macros unless you are using the struct!
        fits_read_key(s->fitsfiles[ii], TINT, "STT_IMJD", &IMJD, comment, &status);
        s->start_MJD[ii] = (long double) IMJD;
        fits_read_key(s->fitsfiles[ii], TINT, "STT_SMJD", &SMJD, comment, &status);
        fits_read_key(s->fitsfiles[ii], TDOUBLE, "STT_OFFS", &OFFS, comment, &status);
        s->start_MJD[ii] += ((long double) SMJD + (long double) OFFS) / SECPERDAY;

        // Are we tracking?
        fits_read_key(s->fitsfiles[ii], TSTRING, "TRK_MODE", ctmp, comment, &status);
        itmp = (strcmp("TRACK", ctmp)==0) ? 1 : 0;
        if (ii==0) s->tracking = itmp;
        else if (s->tracking != itmp)
            printf("Warning!:  TRK_MODE values don't match for files 0 and %d!\n", ii);

        // Now switch to the SUBINT HDU header
        fits_movnam_hdu(s->fitsfiles[ii], BINARY_TBL, "SUBINT", 0, &status);
        get_hdr_double("TBIN", s->dt);
        get_hdr_int("NCHAN", s->num_channels);
        get_hdr_int("NPOL", s->num_polns);
        get_hdr_string("POL_TYPE", s->poln_order);
        fits_read_key(s->fitsfiles[ii], TINT, "NCHNOFFS", &itmp, comment, &status);
        if (itmp > 0)
            printf("Warning!:  First freq channel is not 0 in file %d!\n", ii);
        get_hdr_int("NSBLK", s->spectra_per_subint);
        get_hdr_int("NBITS", s->bits_per_sample);
        fits_read_key(s->fitsfiles[ii], TINT, "NAXIS2", 
                      &(s->num_subint[ii]), comment, &status);
        fits_read_key(s->fitsfiles[ii], TINT, "NSUBOFFS", 
                      &(s->start_subint[ii]), comment, &status);
        s->time_per_subint = s->dt * s->spectra_per_subint;

        // Get the time offset column info and the offset for the 1st row
        {
            double offs_sub;
            int colnum, anynull, numrows;

            // Identify the OFFS_SUB column number
            fits_get_colnum(s->fitsfiles[ii], 0, "OFFS_SUB", &colnum, &status);
            if (status==COL_NOT_FOUND) {
                printf("Warning!:  Can't find the OFFS_SUB column!\n");
                status = 0; // Reset status
            } else {
                if (ii==0) {
                    s->offs_sub_col = colnum;
                } else if (colnum != s->offs_sub_col) {
                    printf("Warning!:  OFFS_SUB column changes between files!\n");
                }
            }

            // Read the OFFS_SUB column value for the 1st row
            fits_read_col(s->fitsfiles[ii], TDOUBLE,
                          s->offs_sub_col, 1L, 1L, 1L,
                          0, &offs_sub, &anynull, &status);

            numrows = (int)((offs_sub - 0.5 * s->time_per_subint) /
                            s->time_per_subint + 1e-7);
            // Check to see if any rows have been deleted or are missing
            if (numrows > s->start_subint[ii]) {
                printf("Warning!:  NSUBOFFS reports %d previous rows\n"
                       "           but OFFS_SUB implies %d.  Using OFFS_SUB.\n"
                       "           Will likely be able to correct for this.\n",
                       s->start_subint[ii], numrows);
            }
            s->start_subint[ii] = numrows;
        }

        // This is the MJD offset based on the starting subint number
        MJDf = (s->time_per_subint * s->start_subint[ii]) / SECPERDAY;
        // The start_MJD values should always be correct
        s->start_MJD[ii] += MJDf;

        // Compute the starting spectra from the times
        MJDf = s->start_MJD[ii] - s->start_MJD[0];
        if (MJDf < 0.0) {
            fprintf(stderr, "Error!: File %d seems to be from before file 0!\n", ii); 
            exit(1);
        }
        s->start_spec[ii] = (long long)(MJDf * SECPERDAY / s->dt + 0.5);

        // Now pull stuff from the other columns
        {
            float ftmp;
            long repeat, width;
            int colnum, anynull;
            
            // Identify the data column and the data type
            fits_get_colnum(s->fitsfiles[ii], 0, "DATA", &colnum, &status);
            if (status==COL_NOT_FOUND) {
                printf("Warning!:  Can't find the DATA column!\n");
                status = 0; // Reset status
            } else {
                if (ii==0) {
                    s->data_col = colnum;
                    fits_get_coltype(s->fitsfiles[ii], colnum, &(s->FITS_typecode), 
                                     &repeat, &width, &status);
                    // This makes CFITSIO treat 1-bit data as written in 'B' mode
                    // even if it was written in 'X' mode originally.  This means
                    // that we unpack it ourselves.
                    if (s->bits_per_sample==1 && s->FITS_typecode==1) {
                        s->FITS_typecode = 11;
                    }
                } else if (colnum != s->data_col) {
                    printf("Warning!:  DATA column changes between files!\n");
                }
            }
            
            // Telescope azimuth
            fits_get_colnum(s->fitsfiles[ii], 0, "TEL_AZ", &colnum, &status);
            if (status==COL_NOT_FOUND) {
                s->azimuth = 0.0;
                status = 0; // Reset status
            } else {
                fits_read_col(s->fitsfiles[ii], TFLOAT, colnum, 
                              1L, 1L, 1L, 0, &ftmp, &anynull, &status);
                if (ii==0) s->azimuth = (double) ftmp;
            }
            
            // Telescope zenith angle
            fits_get_colnum(s->fitsfiles[ii], 0, "TEL_ZEN", &colnum, &status);
            if (status==COL_NOT_FOUND) {
                s->zenith_ang = 0.0;
                status = 0; // Reset status
            } else {
                fits_read_col(s->fitsfiles[ii], TFLOAT, colnum, 
                              1L, 1L, 1L, 0, &ftmp, &anynull, &status);
                if (ii==0) s->zenith_ang = (double) ftmp;
            }
            
            // Observing frequencies
            fits_get_colnum(s->fitsfiles[ii], 0, "DAT_FREQ", &colnum, &status);
            if (status==COL_NOT_FOUND) {
                printf("Warning!:  Can't find the channel freq column!\n");
                status = 0; // Reset status
            } else {
                int jj;
                float *freqs = (float *)malloc(sizeof(float) * s->num_channels);
                fits_read_col(s->fitsfiles[ii], TFLOAT, colnum, 1L, 1L, 
                              s->num_channels, 0, freqs, &anynull, &status);
                
                if (ii==0) {
                    int trigger=0;
                    s->df = ((double)freqs[s->num_channels-1]-
                             (double)freqs[0])/(double)(s->num_channels-1);
                    s->lo_freq = freqs[0];
                    s->hi_freq = freqs[s->num_channels-1];
                    // Now check that the channel spacing is the same throughout
                    for (jj = 0 ; jj < s->num_channels - 1 ; jj++) {
                        ftmp = freqs[jj+1] - freqs[jj];
                        if ((fabs(ftmp - s->df) > 1e-7) && !trigger) {
                            trigger = 1;
                            printf("Warning!:  Channel spacing changes in file %d!\n", ii);
                        }
                    }
                } else {
                    ftmp = fabs(s->df-(freqs[1]-freqs[0]));
                    if (ftmp > 1e-7)
                        printf("Warning!:  Channel spacing changes between files!\n");
                    ftmp = fabs(s->lo_freq-freqs[0]);
                    if (ftmp > 1e-7)
                        printf("Warning!:  Low channel changes between files!\n");
                    ftmp = fabs(s->hi_freq-freqs[s->num_channels-1]);
                    if (ftmp > 1e-7)
                        printf("Warning!:  High channel changes between files!\n");
                }
                free(freqs);
            }
        }
        
        // Compute the samples per file and the amount of padding
        // that the _previous_ file has
        s->num_pad[ii] = 0;
        s->num_spec[ii] = s->spectra_per_subint * s->num_subint[ii];
        if (ii > 0) {
            if (s->start_spec[ii] > s->N) { // Need padding
                s->num_pad[ii-1] = s->start_spec[ii] - s->N;
                s->N += s->num_pad[ii-1];
            }
        }
        s->N += s->num_spec[ii];
    }

    // Convert the position strings into degrees
    /*{
        int d, h, m;
        double sec;
        ra_dec_from_string(s->ra_str, &h, &m, &sec);
        s->ra2000 = hms2rad(h, m, sec) * RADTODEG;
        ra_dec_from_string(s->dec_str, &d, &m, &sec);
        s->dec2000 = dms2rad(d, m, sec) * RADTODEG;
    }*/

    // Are the polarizations summed?
    if ((strncmp("AA+BB", s->poln_order, 5)==0) ||
        (strncmp("INTEN", s->poln_order, 5)==0))
        s->summed_polns = 1;
    else
        s->summed_polns = 0;

    // Is the data IQUV and the user poln is not set?
    if ((strncmp("IQUV", s->poln_order, 4)==0) &&
        (s->use_poln==0))
        s->use_poln = 1; // 1st poln = I
        
    // Calculate some others
    s->T = s->N * s->dt;
    s->orig_df /= (double) s->orig_num_chan;
    s->samples_per_spectra = s->num_polns * s->num_channels;
    // Note:  the following is the number of bytes that will be in
    //        the returned array from CFITSIO, possibly after processing by
    //        us given that we turn 1-, 2-, and 4-bit data into bytes
    //        immediately if bits_per_sample < 8
    if (s->bits_per_sample < 8)
        s->bytes_per_spectra = s->samples_per_spectra;
    else
        s->bytes_per_spectra = (s->bits_per_sample * s->samples_per_spectra) / 8;
    s->samples_per_subint = s->samples_per_spectra * s->spectra_per_subint;
    s->bytes_per_subint = s->bytes_per_spectra * s->spectra_per_subint;
    
    // Compute the bandwidth
    s->BW = s->num_channels * s->df;

    // Flip the bytes for Parkes FB_1BIT data
    if (s->bits_per_sample==1 &&
        strcmp(s->telescope, "Parkes")==0 &&
        strcmp(s->backend, "FB_1BIT")==0) {
        printf("Flipping bit ordering since Parkes FB_1BIT data.\n");
        s->flip_bytes = 1;
    } else {
        s->flip_bytes = 0;
    }

    // Allocate the buffers
    cdatabuffer = gen_bvect(s->bytes_per_subint);
    // Following is twice as big because we use it as a ringbuffer too
    fdatabuffer = gen_fvect(2 * s->spectra_per_subint * s->num_channels);
    s->padvals = gen_fvect(s->num_channels);
    for (ii = 0 ; ii < s->num_channels ; ii++)
        s->padvals[ii] = 0.0;
    offsets = gen_fvect(s->num_channels * s->num_polns);
    scales = gen_fvect(s->num_channels * s->num_polns);
    weights = gen_fvect(s->num_channels);
    // Initialize these if we won't be reading them from the file
    if (s->apply_offset==0) 
        for (ii = 0 ; ii < s->num_channels * s->num_polns ; ii++)
            offsets[ii] = 0.0;
    if (s->apply_scale==0)
        for (ii = 0 ; ii < s->num_channels * s->num_polns ; ii++)
            scales[ii] = 1.0;
    if (s->apply_weight==0)
        for (ii = 0 ; ii < s->num_channels ; ii++)
            weights[ii] = 1.0;
}


