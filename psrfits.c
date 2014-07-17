#include <sys/types.h>
#include <pwd.h>
#include "presto.h"
#include "mask.h"
#include "psrfits.h"
#include <ctype.h>

#define DEBUG_OUT 1
#define strMove(d,s) memmove(d,s,strlen(s)+1)

static unsigned char* cdatabuffer;
static int cur_file = 0, cur_subint = 1, numbuffered = 0;
static long long cur_spec = 0, new_spec = 0;

void get_PSRFITS_subint(float *fdata, unsigned char *cdata, 
                        struct spectra_info *s);

static int mtab[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

double slaCldj(int iy, int im, int id, int *j)
{
   double mjd = 0.0;
   /*  Month lengths in days */

   /* Preset status */
   *j = 0;
   /* Validate year */
   if (iy < -4699) {
      *j = 1;
   } else {
      /* Validate month */
      if (im >= 1 && im <= 12) {
         /* Allow for leap year */
         if (iy % 4 == 0) {
            mtab[1] = 29;
         } else {
            mtab[1] = 28;
         }
         if (iy % 100 == 0 && iy % 400 != 0) {
            mtab[1] = 28;
         }
         /* Validate day */
         if (id < 1 || id > mtab[im - 1]) {
            *j = 3;
         }
         /* Modified Julian Date */
         mjd = (double) ((iy - (12 - im) / 10 + 4712) * 1461 / 4 +
                         ((im + 9) % 12 * 306 + 5) / 10 -
                         (iy - (12 - im) / 10 + 4900) / 100 * 3 / 4 + id - 2399904);
      } else {                  /* Bad month */
         *j = 2;
      }
   }
   return mjd;
}

double hms2rad(int hour, int min, double sec)
/* Convert hours, minutes, and seconds of arc to radians */
{
   return SEC2RAD * (60.0 * (60.0 * (double) hour + (double) min) + sec);
}

double dms2rad(int deg, int min, double sec)
/* Convert degrees, minutes, and seconds of arc to radians */
{
   double sign = 1.0;

   if (deg < 0) sign = -1.0;
   if (deg==0 && (min < 0 || sec < 0.0)) sign = -1.0;
   return sign * ARCSEC2RAD * (60.0 * (60.0 * (double) abs(deg)
                                       + (double) abs(min)) + fabs(sec));
}

char *rmtrail(char *str)
/* Removes trailing space from a string */
{
    int i;
    
    if (str && 0 != (i = strlen(str))) {
        while (--i >= 0) {
            if (!isspace(str[i]))
                break;
        }
        str[++i] = '\0';
    }
    return str;
}

char *rmlead(char *str)
/* Removes leading space from a string */
{
    char *obuf;
    
    if (str) {
        for (obuf = str; *obuf && isspace(*obuf); ++obuf);
        if (str != obuf)
            strMove(str, obuf);
    }
    return str;
}

char *remove_whitespace(char *str)
/* Remove leading and trailing space from a string */
{
    return rmlead(rmtrail(str));
}

void ra_dec_from_string(char *radec, int *h_or_d, int *m, double *s)
/* Return a values for hours or degrees, minutes and seconds        */
/* given a properly formatted RA or DEC string.                     */
/*   radec is a string with J2000 RA  in the format 'hh:mm:ss.ssss' */
/*   or a string with J2000 DEC in the format 'dd:mm:ss.ssss'       */
{
    radec = remove_whitespace(radec);
    sscanf(radec, "%d:%d:%lf\n", h_or_d, m, s);
    if (radec[0]=='-' && *h_or_d==0) {
        *m = -*m;
        *s = -*s;
    }
}

double DATEOBS_to_MJD(char *dateobs, int *mjd_day, double *mjd_fracday)
// Convert DATE-OBS string from PSRFITS primary HDU to a MJD
{
   int year, month, day, hour, min, err;
   double sec;

   sscanf(dateobs, "%4d-%2d-%2dT%2d:%2d:%lf", 
          &year, &month, &day, &hour, &min, &sec);
   *mjd_fracday = (hour + (min + (sec / 60.0)) / 60.0) / 24.0;
   *mjd_day = slaCldj(year, month, day, &err);
   return *mjd_day + *mjd_fracday;
}

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
    s->get_rawblock = &get_PSRFITS_rawblock;
    s->offset_to_spectra = &offset_to_PSRFITS_spectra;

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
    {
        int d, h, m;
        double sec;
        ra_dec_from_string(s->ra_str, &h, &m, &sec);
        s->ra2000 = hms2rad(h, m, sec) * RADTODEG;
        ra_dec_from_string(s->dec_str, &d, &m, &sec);
        s->dec2000 = dms2rad(d, m, sec) * RADTODEG;
    }

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
}


long long offset_to_PSRFITS_spectra(long long specnum, struct spectra_info *s)
// This routine offsets into the PSRFITS files to the spectra
// 'specnum'.  It returns the current spectra number.
{
    int filenum = 0;

    if (specnum > s->N) {
        fprintf(stderr, "Error:  offset spectra %lld is > total spectra %lld\n\n",
               specnum, s->N);
        exit(1);
    }

    // Find which file we need
    while (filenum+1 < s->num_files && specnum > s->start_spec[filenum+1])
        filenum++;

    // Shift to that file
    cur_spec = specnum;
    new_spec = specnum;
    cur_file = filenum;
    numbuffered = 0;

    // Are we in a padding zone?
    if (specnum > (s->start_spec[cur_file] + s->num_spec[cur_file])) {
        // "Seek" to the end of the file
        cur_subint = s->num_subint[cur_file] + 1;
        new_spec = s->start_spec[cur_file+1];
        return specnum;
    }

    // Otherwise, "seek" to the spectra (really a whole subint)
    // Check to make sure that specnum is the start of a subint
    if ((specnum - s->start_spec[cur_file]) % s->spectra_per_subint) {
        fprintf(stderr, "Error:  requested spectra %lld is not the start of a PSRFITS subint\n\n",
                specnum);
        exit(1);
    }
    // Remember zero offset for CFITSIO...
    cur_subint = (specnum - s->start_spec[cur_file]) / s->spectra_per_subint + 1;
    // printf("Current spectra = %lld, subint = %d, in file %d\n", cur_spec, cur_subint, cur_file);
    return specnum;
}


int get_PSRFITS_rawblock(float *fdata, struct spectra_info *s, int *padding)
// This routine reads a single block (i.e subint) from the input files
// which contain raw data in PSRFITS format.  If padding is
// returned as 1, then padding was added and statistics should not be
// calculated.  Return 1 on success.
{
    int numtoread, status = 0, anynull;
    float *fdataptr = fdata;
    
    fdataptr = fdata + numbuffered * s->num_channels;
    // numtoread is always this size since we need to read
    // full PSRFITS subints...
    numtoread = s->spectra_per_subint;
    
    // If our buffer array is offset from last time, 
    // copy the previously offset part into the beginning.
    // New data comes after the old data in the buffer.
    if (numbuffered)
        memcpy((char *)fdata, (char *)(fdata + numtoread * s->num_channels), 
               numbuffered * s->num_channels * sizeof(float));
    
    // Make sure our current file number is valid
    if (cur_file >= s->num_files) return 0;
    
    // Read a subint of data from the DATA col
    if (cur_subint <= s->num_subint[cur_file]) {
        double offs_sub = 0.0;
        // Read the OFFS_SUB column value in case there were dropped blocks
        fits_read_col(s->fitsfiles[cur_file], TDOUBLE, 
                      s->offs_sub_col, cur_subint, 1L, 1L, 
                      0, &offs_sub, &anynull, &status);
        // Set new_spec to proper value, accounting for possibly
        // missing initial rows of data and/or combining observations
        // Note: need to remove start_subint because that was already put
        // into start_spec.  This is important if initial rows are gone.
        new_spec = s->start_spec[cur_file] +
            roundl((offs_sub - (s->start_subint[cur_file] + 0.5)
                    * s->time_per_subint) / s->dt);

        //printf("cur/new_spec = %lld, %lld  s->start_spec[cur_file] = %lld\n", 
        //       cur_spec, new_spec, s->start_spec[cur_file]);
    
        // The following determines if there were lost blocks, or if
        // we are putting different observations together so that
        // the blocks are not aligned
        if (new_spec == cur_spec + numbuffered) {
            // if things look good, with no missing blocks, read the data
            get_PSRFITS_subint(fdataptr, cdatabuffer, s);
            cur_subint++;
            goto return_block;
        }
    } else {
        // We are going to move to the next file, so update
        // new_spec to be the starting spectra from the next file
        // so we can see if any padding is necessary
        if (cur_file < s->num_files-1)
            new_spec = s->start_spec[cur_file+1];
        else
            new_spec = cur_spec + numbuffered;
    }
    
    if (new_spec == cur_spec + numbuffered) {
        // No padding is necessary, so switch files
        cur_file++;
        cur_subint = 1;
        return get_PSRFITS_rawblock(fdata, s, padding);
    }
    
return_block:
    // Apply the corrections that need a full block

    // Increment our static counter (to determine how much data we
    // have written on the fly).
    cur_spec += s->spectra_per_subint;

    return 1;
}


void get_PSRFITS_subint(float *fdata, unsigned char *cdata, 
                        struct spectra_info *s)
{
    float *fptr;
    unsigned char *cptr;
    short *sptr;
    float *ftptr;
    int ii, jj, status = 0, anynull;
    int numtoread = s->samples_per_subint;
    
    // The following allows us to read byte-packed data
    numtoread = s->samples_per_subint * s->bits_per_sample / 8;

    // Now actually read the subint into the temporary buffer
    fits_read_col(s->fitsfiles[cur_file], s->FITS_typecode, 
                  s->data_col, cur_subint, 1L, numtoread, 
                  0, cdata, &anynull, &status);

    if (status) {
        fprintf(stderr, "Error!:  Problem reading record from PSRFITS data file\n"
                "\tfilename = '%s', subint = %d.  FITS status = %d.  Exiting.\n",
                s->filenames[cur_file], cur_subint, status);
        exit(1);
    } //Error handling

    // The following converts that byte-packed data into bytes, in place
    if (s->bits_per_sample == 4) {
        unsigned char uctmp;
        for (ii = numtoread - 1, jj = 2 * numtoread - 1 ; 
             ii >= 0 ; ii--, jj -= 2) {
            uctmp = (unsigned char)cdata[ii];
            cdata[jj] = uctmp & 0x0F;
            cdata[jj-1] = uctmp >> 4;
        }
    } else if (s->bits_per_sample == 2) {
        unsigned char uctmp;
        for (ii = numtoread - 1, jj = 4 * numtoread - 1 ; 
             ii >= 0 ; ii--, jj -= 4) {
            uctmp = (unsigned char)cdata[ii];
            cdata[jj] = (uctmp & 0x03);
            cdata[jj-1] = ((uctmp >> 0x02) & 0x03);
            cdata[jj-2] = ((uctmp >> 0x04) & 0x03);
            cdata[jj-3] = ((uctmp >> 0x06) & 0x03);
        }
    } else if (s->bits_per_sample == 1) {
        unsigned char uctmp;
        for (ii = numtoread - 1, jj = 8 * numtoread - 1 ; 
             ii >= 0 ; ii--, jj -= 8) {
            uctmp = (unsigned char)cdata[ii];
            cdata[jj] = (uctmp & 0x01);
            cdata[jj-1] = ((uctmp >> 0x01) & 0x01);
            cdata[jj-2] = ((uctmp >> 0x02) & 0x01);
            cdata[jj-3] = ((uctmp >> 0x03) & 0x01);
            cdata[jj-4] = ((uctmp >> 0x04) & 0x01);
            cdata[jj-5] = ((uctmp >> 0x05) & 0x01);
            cdata[jj-6] = ((uctmp >> 0x06) & 0x01);
            cdata[jj-7] = ((uctmp >> 0x07) & 0x01);
        }
    }

    if (s->bits_per_sample == 1 && s->flip_bytes) {
        // Hack to flip each byte of data if needed
        int offset;
        unsigned char uctmp;
        for (ii = 0 ; ii < s->bytes_per_subint/8 ; ii++) {
            offset = ii * 8;
            for (jj = 0 ; jj < 4 ; jj++) {
                uctmp = cdata[offset+jj];
                cdata[offset+jj] = cdata[offset+8-1-jj];
                cdata[offset+8-1-jj] = uctmp;
            }
        }
    }

    // Now convert all of the data into floats

    // The following allows us to work with single polns out of many
    // or to sum polarizations if required
    if (s->num_polns > 1) {
        int sum_polns = 0;
        
        if ((0==strncmp(s->poln_order, "AABB", 4)) || 
            (s->num_polns == 2)) 
            sum_polns = 1;
        // User chose which poln to use
        if (s->use_poln > 0 || ((s->num_polns > 2) && !sum_polns)) {
            int idx;
            fptr = fdata;
            if (s->bits_per_sample==16) {
                for (ii = 0 ; ii < s->spectra_per_subint ; ii++) {
                    idx = (s->use_poln-1) * s->num_channels;
                    sptr = (short *)cdata + ii * s->samples_per_spectra + idx;
                    for (jj = 0 ; jj < s->num_channels ; jj++)
                        *fptr++ = *sptr++;
                }
            } else if (s->bits_per_sample==32) {
                for (ii = 0 ; ii < s->spectra_per_subint ; ii++) {
                    idx = (s->use_poln-1) * s->num_channels;
                    ftptr = (float *)cdata + ii * s->samples_per_spectra + idx;
                    for (jj = 0 ; jj < s->num_channels ; jj++)
                        *fptr++ = *ftptr++;
                }
            } else {
                for (ii = 0 ; ii < s->spectra_per_subint ; ii++) {
                    idx = (s->use_poln-1) * s->num_channels;
                    cptr = cdata + ii * s->samples_per_spectra + idx;
                    for (jj = 0 ; jj < s->num_channels ; jj++)
                        *fptr++ = *cptr++;
                }
            }
        } else if (sum_polns) { // sum the polns if there are 2 by default
            int idx = s->num_channels;
            fptr = fdata;
            if (s->bits_per_sample==16) {
                for (ii = 0 ; ii < s->spectra_per_subint ; ii++) {
                    sptr = (short *)cdata + ii * s->samples_per_spectra;
                    for (jj = 0 ; jj < s->num_channels ; jj++, sptr++, fptr++) {
                        *fptr = *sptr;
                        *fptr += *(sptr+idx);
                    }
                }
            } else if (s->bits_per_sample==32) {
                for (ii = 0 ; ii < s->spectra_per_subint ; ii++) {
                    ftptr = (float *)cdata + ii * s->samples_per_spectra;
                    for (jj = 0 ; jj < s->num_channels ; jj++, ftptr++, fptr++) {
                        *fptr = *ftptr;
                        *fptr += *(ftptr+idx);
                    }	
                }
            } else {
                for (ii = 0 ; ii < s->spectra_per_subint ; ii++) {
                    cptr = cdata + ii * s->samples_per_spectra;
                    for (jj = 0 ; jj < s->num_channels ; jj++, cptr++, fptr++) {
                        *fptr = *cptr;
                        *fptr += *(cptr+idx);
                    }
                }
            }
        }
    } else {  // This is for normal single-polarization data
        fptr = fdata;
        if (s->bits_per_sample==16) {
            for (ii = 0 ; ii < s->spectra_per_subint ; ii++) {
                sptr = (short *)cdata + ii * s->samples_per_spectra;
                for (jj = 0 ; jj < s->num_channels ; jj++)
                    *fptr++ = *sptr++;
            }
        } else {
            for (ii = 0 ; ii < s->spectra_per_subint ; ii++) {
                cptr = cdata + ii * s->samples_per_spectra;
                for (jj = 0 ; jj < s->num_channels ; jj++)
                    *fptr++ = *cptr++;
            }
        }
    }
}
