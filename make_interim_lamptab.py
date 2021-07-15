import numpy as np
import calcos
import glob
import os
import shutil
from datetime import datetime
from astropy.io import fits
from astropy.table import Table, vstack
#from add_lamptab_rows import *
from updating_lamptab import *
from common_correlation_tasks import *
from matplotlib import pyplot as plt

# -------------------------------------------------------------------------------
def prepare_correlation(cen, fp, segment, lamptab):
    '''
    Grabbing the reference and flux data from the lamptab. This is used in
    the find_fp_pix_shift function.

    Inputs
    ------
    cen : int
      current cenwave we are trying to update
    fp : int
      current FPPOS (1, 2, or 4) that we are updating.
    segment : str
      current segment we are trying to update
    lamptab : str
      filename of the LAMPTAB for which you want the FP_PIXEL_SHIFT keyword to
      be updated; presumably the one created from Elaine's code

    Outputs
    -------
    ref_flux : arr
      the intensity column of the lamptab at FPPOS=3 for the specified cenwave
      and segment
    other_flux : arr
      the intensity column of the lamptab at the specified cenwave, segment, and
      fppos
    '''

    with fits.open(lamptab) as lamp:
        # reference file (FPPOS 3)
        wh_0 = np.where((lamp[1].data['CENWAVE'] == cen) &
                        (lamp[1].data['FPOFFSET'] == 0) &
                        (lamp[1].data['SEGMENT'] == segment))[0]
        # other file (FPPOS 1, 2, or 4)
        wh_other = np.where((lamp[1].data['CENWAVE'] == cen) &
                            (lamp[1].data['FPOFFSET'] == fp) &
                            (lamp[1].data['SEGMENT'] == segment))[0]

        if not len(wh_0) or not len(wh_other):
            # print('No data for {}/fp{}/{}'.format(cen, fp, segment))
            return [], []

        ref_flux = lamp[1].data['INTENSITY'][wh_0][0]
        other_flux = lamp[1].data['INTENSITY'][wh_other][0]
        print(ref_flux.max(), other_flux.max())
    return ref_flux, other_flux


# --------------------------------

def find_fp_pix_shift(lamptab, outfile, lp):
    ''' Finds the FP_PIXEL_SHIFT column for the lamptab by cross-correlating
    each FPPOS 1, 2, and 4 to FPPOS 3 for the given LAMPTAB

    Inputs
    ------
    lamptab : str
      filename of the LAMPTAB for which you want the FP_PIXEL_SHIFT keyword to
      be updated; presumably the one created from Elaine's code
    outfile : str
       filename of the text file to be saved containing the FP_PIXEL_SHIFTs

    Outputs
    -------
    None
    '''

    with fits.open(lamptab) as lamp:
        cenwaves = np.unique(lamp[1].data['CENWAVE'])
        segments = np.unique(lamp[1].data['SEGMENT'])
        fps = np.unique(lamp[1].data['FPOFFSET'])

    correlation_data = []

    for cenwave in cenwaves:
        print(cenwave)
        #if cenwave == 1533 or cenwave == 800:
            # continue # not updating the current modes
        if lp == 3:
            if cenwave == 800:
                for fpoff in fps:

                    if fpoff == 0:
                        continue  # FPPOS 3 is the zero-point
                    for seg in segments:
                        ref_flux, other_flux = prepare_correlation(cenwave, fpoff, seg, lamptab)
                        # print (cenwave,fpoff,seg)
                        if not len(ref_flux):
                            if cenwave == 800 and seg == 'FUVB':
                                # print('800 FUVB Shift: {}'.format(shift))
                                print('800 FUVB no data to shift')
                                # There is no 800 FUVB setting
                                continue
                            else:
                                raise ValueError('No flux in {}/{}/{}'.format(cenwave, fpoff, seg))

                            # if cenwave == 1280 and seg == 'FUVB':
                            # We want the FP_PIXEL_SHIFT copied over from FUVA
                            # pass through so the shift stays the same as FUVA
                            #   print('1280 FUVB Shift: {}'.format(shift))
                            #  pass
                        else:
                            x_pixels = np.arange(len(ref_flux))
                            plt.plot(ref_flux, color='red')
                            plt.plot(other_flux, color='blue')
                            plt.show()
                            shift, cc = cross_correlate(ref_flux, other_flux, wave_a=x_pixels,
                                                       wave_b=x_pixels, subsample=1)
                            shift *= -1
                            print('CORRELATING {} vs {} Shift: {}'.format(0, fpoff, shift))

                            correlation_data.append((0,
                                                 fpoff,
                                                 cenwave,
                                                 seg,
                                                 shift,
                                                 cc))

                            names = ('ref_fp',
                                     'obs_fp',
                                     'cenwave',
                                     'segment',
                                     'fp_shift',
                                     'correlation_coeff')

                            if not len(correlation_data):
                                print('no data for target', len(correlation_data))
                                return Table(names=names)

                            out_table = Table(rows=correlation_data, names=names)
                            if len(out_table):
                                correlation_table = vstack(out_table)
                                correlation_table.write(outfile, format='ascii.csv')
                                print('saved {}'.format(outfile))
                                os.chmod(outfile, 0o777)
        elif lp == 5:
            if cenwave == 1291 or cenwave == 1300 or cenwave == 1309 or cenwave == 1318 or cenwave == 1327:                    
                for fpoff in fps:

                    if fpoff == 0:
                        continue  # FPPOS 3 is the zero-point
                    for seg in segments:
                        ref_flux, other_flux = prepare_correlation(cenwave, fpoff, seg, lamptab)
                        # print (cenwave,fpoff,seg)
                        if not len(ref_flux):
                            if cenwave == 800 and seg == 'FUVB':
                                # print('800 FUVB Shift: {}'.format(shift))
                                print('800 FUVB no data to shift')
                                # There is no 800 FUVB setting
                                continue
                            else:
                                raise ValueError('No flux in {}/{}/{}'.format(cenwave, fpoff, seg))

                            # if cenwave == 1280 and seg == 'FUVB':
                            # We want the FP_PIXEL_SHIFT copied over from FUVA
                            # pass through so the shift stays the same as FUVA
                            #   print('1280 FUVB Shift: {}'.format(shift))
                            #  pass
                        else:
                            x_pixels = np.arange(len(ref_flux))
                            plt.plot(ref_flux, color='red')
                            plt.plot(other_flux, color='blue')
                            plt.show()
                            shift, cc = cross_correlate(ref_flux, other_flux, wave_a=x_pixels,
                                                       wave_b=x_pixels, subsample=1)
                            shift *= -1
                            print('CORRELATING {} vs {} Shift: {}'.format(0, fpoff, shift))

                            correlation_data.append((0,
                                                 fpoff,
                                                 cenwave,
                                                 seg,
                                                 shift,
                                                 cc))

                            names = ('ref_fp',
                                     'obs_fp',
                                     'cenwave',
                                     'segment',
                                     'fp_shift',
                                     'correlation_coeff')

                            if not len(correlation_data):
                                print('no data for target', len(correlation_data))
                                return Table(names=names)

                            out_table = Table(rows=correlation_data, names=names)
                            if len(out_table):
                                correlation_table = vstack(out_table)
                                correlation_table.write(outfile, format='ascii.csv')
                                print('saved {}'.format(outfile))
                                os.chmod(outfile, 0o777)

# ------------------------------------------------------------------------------------
def update_lamptab(new_shifts_file, updated_lamptab):
    '''Updating LAMPTAB with the new FP_PIXEL_SHIFT values from a txt
       file created earlier in this script.

    Inputs
    ------
    new_shifts_file : str
       The name of a text file that contains the cross-correlation results from
       correlating different FPPOS to FPPOS 3 for every cenwave and segment.
       This is a result from find_fp_pix_shift.
    updated_lamptab : str
       The name of the LAMPTAB you want to update (Most likely the output from
       Elaine's code).

    Outputs
    -------
    None
    '''

    shifts_table = ascii.read(new_shifts_file)
    cenwaves = np.unique(shifts_table['cenwave'])
    fpoff = np.unique(shifts_table['obs_fp'])
    segments = np.unique(shifts_table['segment'])

    for cen in cenwaves:
        #if cen == 1533 or cen == 800:
            #print(cen)
        for fp in fpoff:
            for segment in segments:
                # There should only be one of these values in the table
                wh_new_shift = np.where((shifts_table['cenwave'] == cen) &
                                        (shifts_table['obs_fp'] == fp) &
                                        (shifts_table['segment'] == segment))[0]
                if not len(wh_new_shift):
                    # 1105 FUVB is skipped
                    print('No shift for {}/{}/{}'.format(cen, fp, segment))
                    continue

                with fits.open(updated_lamptab, mode="update") as hdu:
                    wh_lamp = np.where((hdu[1].data['CENWAVE'] == cen) &
                                       (hdu[1].data['FPOFFSET'] == fp) &
                                       (hdu[1].data['SEGMENT'] == segment))[0][0]
                    applied_shift = shifts_table['fp_shift'][wh_new_shift[0]]
                    hdu[1].data[wh_lamp]['FP_PIXEL_SHIFT'] = applied_shift

                with fits.open(updated_lamptab) as hdu:
                    print(cen, fp, segment, applied_shift)


# -------------------------------------------------------------------------------
if __name__ == '__main__':
    # current_lamptab = '/grp/hst/cdbs/lref/24915198l_lamp.fits'
    # version=50
    #    make_lamp(current_lamptab, 4, version)

    # Elaine: I think this should be the LP3 lamptab that's in CRDS
    templamptab0 = '2018-06-26_v1_lamp.fits'

    # pointing to directory where calcos is run on corrtags with drift no corrected
    # datadir1='/grp/hst/cos2/c1533/data/15459/corrdir/old_wavecorr_on/'

    datadir2 = '/grp/hst/cos2/c800/data/15484/corrdir/old_wavecorr_on/'

    # x1dfiles1 = glob.glob(os.path.join(datadir1,'*x1d.fits'))
    # changing this to lampflash files so that we use non-drifted data
    # lampflashes1 = glob.glob(os.path.join(datadir1, '*lampflash.fits'))

    # x1dfiles2 = glob.glob(os.path.join(datadir2,'*x1d.fits'))
    lampflashes2 = glob.glob(os.path.join(datadir2, '*lampflash.fits'))

    # x1dfiles=x1dfiles1+x1dfiles2
    lampflashes = lampflashes2  # +lampflashes1

    # version=0
    # Elaine: this is the new output file name?
    templamptab1 = '2018-07-03_v0_lamp.fits'

    # Elaine: this comes from the updating_lamptab.py script
    # it makes a new lamptab file with the intensity column updated
    makelamptemplatefile(templamptab1, templamptab0, lampflashes)
    # previously used to pass x1dfiles instead of lampflashes

    save_file = 'interim_FP_shifts_NOdrift.txt'
    find_fp_pix_shift(templamptab1, save_file)

    # version=2
    # newlamptab='2018-06-26_v{}_lamp.fits'.format(version)

    # this updates the FP_PIXEL_SHIFTs in the new lamptab you just made
    update_lamptab(save_file, templamptab1)
