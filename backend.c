#include <sys/types.h>
#include <pwd.h>
#include <ctype.h>
#include "backend.h"
#include "vectors.h"
#include "makeinf.h"

static long long currentspectra = 0;

#define SWAP(a,b) tmpswap=(a);(a)=(b);(b)=tmpswap;

extern int clip_times(float *rawdata, int ptsperblk, int numchan, 
                      float clip_sigma, float *good_chan_levels);
extern void float_dedisp(float *data, float *lastdata,
                         int numpts, int numchan,
                         int *delays, float approx_mean, float *result);
extern void dedisp_subbands(float *data, float *lastdata,
                            int numpts, int numchan, 
                            int *delays, int numsubbands, float *result);
extern short transpose_float(float *a, int nx, int ny, unsigned char *move, 
                             int move_size);
extern double DATEOBS_to_MJD(char *dateobs, int *mjd_day, double *mjd_fracday);
extern void read_PSRFITS_files(struct spectra_info *s);



void close_rawfiles(struct spectra_info *s)
{
    int ii;
    int status = 0;
    for (ii = 0 ; ii < s->num_files ; ii++)
        fits_close_file(s->fitsfiles[ii], &status);
    free(s->fitsfiles); 
}

void add_padding(float *fdata, float *padding, int numchan, int numtopad)
{
    int ii;
    for (ii = 0; ii < numtopad; ii++)
        memcpy(fdata + ii * numchan, padding, numchan * sizeof(float));
}


void spectra_info_set_defaults(struct spectra_info *s) {
    strcpy(s->telescope, "unset");
    strcpy(s->observer, "unset");
    strcpy(s->source, "unset");
    strcpy(s->frontend, "unset");
    strcpy(s->backend, "unset");
    strcpy(s->project_id, "unset");
    strcpy(s->date_obs, "unset");
    strcpy(s->ra_str, "unset");
    strcpy(s->dec_str, "unset");
    strcpy(s->poln_type, "unset");
    strcpy(s->poln_order, "unset");
    s->datatype = UNSET;
    s->N = 0;
    s->T = 0.0;
    s->dt = 0.0;
    s->fctr = 0.0;
    s->lo_freq = 0.0;
    s->hi_freq = 0.0;
    s->orig_df = 0.0;
    s->chan_dm = 0.0;
    s->df = 0.0;
    s->BW = 0.0;
    s->ra2000 = 0.0;
    s->dec2000 = 0.0;
    s->azimuth = 0.0;
    s->zenith_ang = 0.0;
    s->beam_FWHM = 0.0;
    s->time_per_subint = 0.0;
    s->scan_number = 0;
    s->tracking = 1;
    s->orig_num_chan = 0;
    s->num_channels = 0;
    s->num_polns = 0;
    s->num_beams = 1;
    s->beamnum = 0;
    s->summed_polns = 1;
    s->FITS_typecode = 0;
    s->bits_per_sample = 0;
    s->bytes_per_spectra = 0;
    s->samples_per_spectra = 0;
    s->bytes_per_subint = 0;
    s->spectra_per_subint = 0;
    s->samples_per_subint = 0;
    s->min_spect_per_read = 0;
    s->num_files = 0;
    s->offs_sub_col = 0;
    s->dat_wts_col = 0;
    s->dat_offs_col = 0;
    s->dat_scl_col = 0;
    s->data_col = 0;
    s->apply_scale = 0;
    s->apply_offset = 0;
    s->apply_weight = 0;
    s->apply_flipband = 0;
    s->remove_zerodm = 0;
    s->use_poln = 0;
    s->flip_bytes = 0;
    s->zero_offset = 0.0;
    s->clip_sigma = 0.0;
    s->start_MJD = NULL;
    s->files = NULL;
    s->fitsfiles = NULL;
    s->padvals = NULL;
    s->header_offset = NULL;
    s->start_subint = NULL;
    s->num_subint = NULL;
    s->start_spec = NULL;
    s->num_spec = NULL;
    s->num_pad = NULL;
};


long long offset_to_spectra(long long specnum, struct spectra_info *s)
// This routine offsets into the raw data files to the spectra
// 'specnum'.  It returns the current spectra number.
{
    long long retval;
    retval = s->offset_to_spectra(specnum, s);
    currentspectra = retval;
    return retval;
}


int read_rawblocks(float *fdata, int numsubints, struct spectra_info *s, int *padding)
// This routine reads numsubints rawdata blocks from raw radio pulsar
// data. The floating-point filterbank data is returned in rawdata
// which must have a size of numsubints * s->samples_per_subint.  The
// number of blocks read is returned.  If padding is returned as 1,
// then padding was added and statistics should not be calculated.
{
    int ii, retval = 0, gotblock = 0, pad = 0, numpad = 0, numvals;
    static float *rawdata = NULL;
    static int firsttime = 1;

    numvals = s->spectra_per_subint * s->num_channels;
    if (firsttime) {
        // Needs to be twice as large for buffering if adding observations together
        rawdata = gen_fvect(2 * numvals);
        firsttime = 0;
    }
    *padding = 0;
    for (ii = 0; ii < numsubints; ii++) {
        gotblock = s->get_rawblock(rawdata, s, &pad);
        if (gotblock==0) break;
        retval += gotblock;
        memcpy(fdata + ii * numvals, rawdata, numvals * sizeof(float));
        if (pad) numpad++;
    }
    if (gotblock==0) {  // Now fill the rest of the data with padding
        for (; ii < numsubints; ii++) {
            int jj;
            for (jj = 0; jj < s->spectra_per_subint; jj++)
                memcpy(fdata + ii * numvals + jj * s->num_channels, 
                       s->padvals, s->num_channels * sizeof(float));
        }
        numpad++;
    }

    /* Return padding 'true' if any block was padding */
    if (numpad) *padding = 1;
    return retval;
}
