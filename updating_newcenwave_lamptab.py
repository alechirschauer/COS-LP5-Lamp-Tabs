from astropy.io import fits, ascii
from astropy.table import Table, vstack

import numpy as np
import calcos
import glob
import os
import shutil
from datetime import datetime

from common_correlation_tasks import *

#  Order of what we should do:
#  1. open up new lamptab 
#  2. Edit it so the FP_PIXEL_SHIFT is correct
#  3. Copy and create a new DISPTAB to edit the zero points
#  4. Open the LP1 LAMPTAB and correlate the LP3 and LP1 LAMPTAB to get
#    the new DISPTAB zero-point shifts_table


# -------------------------------------------------------------------------------
def makelamptemplatefile(newfilename, oldfilename, templatefiles):

    old = fits.open(oldfilename)
    olddata = old[1].data

    # one by one, open the new lamp template spectra taken
    # find which column in the old file it corresponds to
    # replace the intensity value in that column with the new data (counts * exptime)

    for newfile in templatefiles:

        with fits.open(newfile) as f:
            data = f[1].data
            hdr0 = f[0].header
            hdr1 = f[1].header

        opt_elem = hdr0['opt_elem']
        cenwave = hdr0['cenwave']
        fpoffset = hdr0['fpoffset']
        exptime = hdr1['exptime']

        print('{0} {1} {2} {3}'.format(opt_elem, cenwave, fpoffset, exptime))

        if exptime > 1080:  # bypass long exposures
            continue

        elif cenwave == 1105:
            segmentA = data['segment'][0]

            correctcolumnA = np.where(
                (olddata['segment'] == segmentA)
                & (olddata['cenwave'] == cenwave)
                & (olddata['opt_elem'] == opt_elem)
                & (olddata['fpoffset'] == fpoffset))

            olddata['intensity'][correctcolumnA] = np.round(data['net'][0] * exptime)

        elif cenwave == 800:  # don't replace segment B of 1280

            segmentA = data['segment'][0]

            correctcolumnA = np.where(
                (olddata['segment'] == segmentA)
                & (olddata['cenwave'] == cenwave)
                & (olddata['opt_elem'] == opt_elem)
                & (olddata['fpoffset'] == fpoffset))

            olddata['intensity'][correctcolumnA] = np.round(data['net'][0] * exptime)

        else:
            segmentA = data['segment'][0]
            segmentB = data['segment'][1]

            correctcolumnA = np.where((olddata['segment'] == segmentA)
                                      & (olddata['cenwave'] == cenwave)
                                      & (olddata['opt_elem'] == opt_elem)
                                      & (olddata['fpoffset'] == fpoffset))

            correctcolumnB = np.where((olddata['segment'] == segmentB)
                                      & (olddata['cenwave'] == cenwave)
                                      & (olddata['opt_elem'] == opt_elem)
                                      & (olddata['fpoffset'] == fpoffset))

            olddata['intensity'][correctcolumnA] = np.round(data['net'][0] * exptime)
            olddata['intensity'][correctcolumnB] = np.round(data['net'][1] * exptime)

    # write the new file
    old.writeto(newfilename, overwrite=True)

    old.close()


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
            #print('No data for {}/fp{}/{}'.format(cen, fp, segment))
            return [], []

        ref_flux = lamp[1].data['INTENSITY'][wh_0][0]
        other_flux = lamp[1].data['INTENSITY'][wh_other][0]

    return ref_flux, other_flux


#--------------------------------

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
        print (cenwave)
        #if cenwave ==  1533 or cenwave == 800:
        if lp == 3:
            if cenwave == 800:
                #continue # not updating the current modes
                for fpoff in fps:
                
                    if fpoff == 0:
                        continue  # FPPOS 3 is the zero-point
                    for seg in segments:
                        ref_flux, other_flux = prepare_correlation(cenwave, fpoff, seg, lamptab)
                        #print (cenwave,fpoff,seg)
                        if not len(ref_flux):
                            if cenwave == 800 and seg == 'FUVB':
                                #print('800 FUVB Shift: {}'.format(shift))
                                print('800 FUVB no data to shift')
                                # There is no 800 FUVB setting
                                continue
                            else:
                                raise ValueError('No flux in {}/{}/{}'.format(cenwave, fpoff, seg))
                        
                            #if cenwave == 1280 and seg == 'FUVB':
                            # We want the FP_PIXEL_SHIFT copied over from FUVA
                            # pass through so the shift stays the same as FUVA
                            #   print('1280 FUVB Shift: {}'.format(shift))
                            #  pass
                        else:
                            x_pixels = np.arange(len(ref_flux))
                        
                            shift, cc = cross_correlate(ref_flux, other_flux, wave_a=x_pixels,
                                                        wave_b=x_pixels, subsample=1)
                            shift *= -1
                            print('CORRELATING {} vs {} Shift: {}'.format(0, fpoff, shift))
                        
                            correlation_data.append((0,
                                                     fpoff,
                                                     cenwave,
                                                     seg,
                                                     shift,
                                                     cc ))
                        
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
                #continue # not updating the current modes
                for fpoff in fps:
                
                    if fpoff == 0:
                        continue  # FPPOS 3 is the zero-point
                    for seg in segments:
                        ref_flux, other_flux = prepare_correlation(cenwave, fpoff, seg, lamptab)
                        #print (cenwave,fpoff,seg)
                        if not len(ref_flux):
                            if cenwave == 800 and seg == 'FUVB':
                                #print('800 FUVB Shift: {}'.format(shift))
                                print('800 FUVB no data to shift')
                                # There is no 800 FUVB setting
                                continue
                            else:
                                raise ValueError('No flux in {}/{}/{}'.format(cenwave, fpoff, seg))
                        
                            #if cenwave == 1280 and seg == 'FUVB':
                            # We want the FP_PIXEL_SHIFT copied over from FUVA
                            # pass through so the shift stays the same as FUVA
                            #   print('1280 FUVB Shift: {}'.format(shift))
                            #  pass
                        else:
                            x_pixels = np.arange(len(ref_flux))
                        
                            shift, cc = cross_correlate(ref_flux, other_flux, wave_a=x_pixels,
                                                        wave_b=x_pixels, subsample=1)
                            shift *= -1
                            print('CORRELATING {} vs {} Shift: {}'.format(0, fpoff, shift))
                        
                            correlation_data.append((0,
                                                     fpoff,
                                                     cenwave,
                                                     seg,
                                                     shift,
                                                     cc ))
                        
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

#-------------------------------------------------------------------------------

def update_lamptab(new_shifts_file, updated_lamptab):
    '''Updating Elaine's LAMPTAB with the new FP_PIXEL_SHIFT values from a txt
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
        for fp in fpoff:
            for segment in segments:
                # There should only be one of these values in the table
                wh_new_shift = np.where( (shifts_table['cenwave'] == cen) &
                                         (shifts_table['obs_fp'] == fp) &
                                         (shifts_table['segment'] == segment))[0]
                if not len(wh_new_shift):
                    # 1105 FUVB is skipped
                    print('No shift for {}/{}/{}'.format(cen, fp, segment))
                    continue

                with fits.open(updated_lamptab, mode="update") as hdu:
                    wh_lamp = np.where( (hdu[1].data['CENWAVE'] == cen) &
                                        (hdu[1].data['FPOFFSET'] == fp) &
                                        (hdu[1].data['SEGMENT'] == segment))[0][0]
                    applied_shift = shifts_table['fp_shift'][wh_new_shift[0]]
                    hdu[1].data[wh_lamp]['FP_PIXEL_SHIFT'] = applied_shift

                with fits.open(updated_lamptab) as hdu:
                    print(cen, fp, segment, applied_shift)


# -------------------------------------------------------------------------------
def update_zeropoint(lamp_new_name, lamp_old_name, new_disp_name, old_disp_name, lp):
    ''' This function updates the DISPTAB by cross-correlating the
        LP1 lamp template to the new lamp template we observed at LP3

    Inputs
    ------
    lamp_new_name : str
        The name of the new LAMPTAB from Elaine's script.
    lamp_name_old : str
        The name of the LP1 LAMPTAB (x6q17585l_lamp.fits).
    new_disp_name : str
        The name you want the new DISPTAB to be.
    old_disp_name : str
        The DISPTAB associated with the LP1 LAMPTAB ; currently this can be
        05i1639ml_disp.fits (LP1 updated), 0bn1606sl_disp.fits (LP2 updated),
        18118285l_disp.fits (includes 1223), or xaa18189l_disp.fits (LP1 original)

    Outputs
    -------
    None
    '''

    with fits.open(new_disp_name) as disp:
        cenwaves = np.unique(disp[1].data['CENWAVE'])
        segments = np.unique(disp[1].data['SEGMENT'])

    print('Updating DISPTAB')
    # will need to add in 1222 and 1223 for LP4 to this list.
    for cenwave in cenwaves:
        #if (cenwave == 1533) or (cenwave == 800):
        if lp == 3:
            if cenwave == 800:
                for segment in segments:

                    if (cenwave == 800) and segment == 'FUVB':
                        # there is no FUVB for 1105
                        break

                    fpoff = 0 # we only care about the difference for FPPOS 3, not averaging the shifts for all FPPOS
                    x_pixels = np.arange(16384)
                    with fits.open(lamp_new_name) as lamp_new:
                        wh_lamp = np.where((lamp_new[1].data['CENWAVE'] == cenwave) &
                                       (lamp_new[1].data['FPOFFSET'] == fpoff) &
                                       (lamp_new[1].data['SEGMENT'] == segment))[0]
                        if not len(wh_lamp):
                            continue
                        new_flux = lamp_new[1].data[wh_lamp][0]['INTENSITY']/ lamp_new[1].data[wh_lamp][0]['INTENSITY'].max()
                        fp_shift_new = lamp_new[1].data[wh_lamp][0]['FP_PIXEL_SHIFT']

                    with fits.open(lamp_old_name) as lamp_old:
                        wh_lamp = np.where((lamp_old[1].data['CENWAVE'] == cenwave) &
                                       (lamp_old[1].data['FPOFFSET'] == fpoff) &
                                      (lamp_old[1].data['SEGMENT'] == segment))[0][0]
                        old_flux = lamp_old[1].data[wh_lamp]['INTENSITY']/lamp_old[1].data[wh_lamp]['INTENSITY'].max()
                        fp_shift_old = lamp_old[1].data[wh_lamp]['FP_PIXEL_SHIFT']

            
            
                        shift, cc = cross_correlate(old_flux, new_flux, wave_a=x_pixels,
                                                wave_b=x_pixels, subsample=1)
                        shift *= -1  # Shift between new LAMPTAB and old LAMPTAB

                        # I think I want to do this to account for the difference in the
                        #  FP_PIXEL_SHIFT correlation that's already been applied
                        cc_shift = (shift-(fp_shift_new-fp_shift_old))

                        if not cc_shift:
                            'No Shift for {} {} {}'.format(cenwave, fpoff, segment)
                            continue
                        with fits.open(old_disp_name) as old_disp:
                            wh_old_disp = np.where((old_disp[1].data['SEGMENT'] == segment) &
                                                   (old_disp[1].data['CENWAVE'] == cenwave) &
                                                   (old_disp[1].data['APERTURE'] == 'PSA'))[0][0]
                            #coeffs_to_update = old_disp[1].data[wh_old_disp]['COEFF']
                            d_to_update = old_disp[1].data[wh_old_disp]['D']

                        with fits.open(new_disp_name, mode='update') as hdu:
                            wh_disp = np.where((hdu[1].data['SEGMENT'] == segment) &
                                        (hdu[1].data['CENWAVE'] == cenwave) &
                                        (hdu[1].data['APERTURE'] == 'PSA'))[0][0]

                            ## Updating the zero-point in the COEFF value, not the D value.
                            #hdu[1].data['COEFF'][wh_disp][0] = coeffs_to_update[0] - (cc_shift*coeffs_to_update[1])
                            hdu[1].data['D'][wh_disp] = d_to_update + cc_shift

                            #print('UPDATING {} {} value={}A difference={}pix {}A'.format(cenwave, segment,
                            #                                                     coeffs_to_update[0] - (cc_shift*coeffs_to_update[1]),
                            #                                                     cc_shift, cc_shift*coeffs_to_update[1]))

                        with fits.open(new_disp_name) as disp:
                            new_d = disp[1].data[wh_disp]['D']
                            print('Updating {}/{}: {} to {} \n Difference of: {}'.format(cenwave, segment, d_to_update, new_d, cc_shift))

        elif lp == 5:
            if cenwave == 1291 or cenwave == 1300 or cenwave == 1309 or cenwave == 1318 or cenwave == 1327:    
                for segment in segments:

                    if (cenwave == 800) and segment == 'FUVB':
                        # there is no FUVB for 1105
                        break

                    fpoff = 0 # we only care about the difference for FPPOS 3, not averaging the shifts for all FPPOS
                    x_pixels = np.arange(16384)
                    with fits.open(lamp_new_name) as lamp_new:
                        wh_lamp = np.where((lamp_new[1].data['CENWAVE'] == cenwave) &
                                       (lamp_new[1].data['FPOFFSET'] == fpoff) &
                                       (lamp_new[1].data['SEGMENT'] == segment))[0]
                        if not len(wh_lamp):
                            continue
                        new_flux = lamp_new[1].data[wh_lamp][0]['INTENSITY']/ lamp_new[1].data[wh_lamp][0]['INTENSITY'].max()
                        fp_shift_new = lamp_new[1].data[wh_lamp][0]['FP_PIXEL_SHIFT']

                    with fits.open(lamp_old_name) as lamp_old:
                        wh_lamp = np.where((lamp_old[1].data['CENWAVE'] == cenwave) &
                                       (lamp_old[1].data['FPOFFSET'] == fpoff) &
                                      (lamp_old[1].data['SEGMENT'] == segment))[0][0]
                        old_flux = lamp_old[1].data[wh_lamp]['INTENSITY']/lamp_old[1].data[wh_lamp]['INTENSITY'].max()
                        fp_shift_old = lamp_old[1].data[wh_lamp]['FP_PIXEL_SHIFT']

            
            
                        shift, cc = cross_correlate(old_flux, new_flux, wave_a=x_pixels,
                                                wave_b=x_pixels, subsample=1)
                        shift *= -1  # Shift between new LAMPTAB and old LAMPTAB

                        # I think I want to do this to account for the difference in the
                        #  FP_PIXEL_SHIFT correlation that's already been applied
                        cc_shift = (shift-(fp_shift_new-fp_shift_old))

                        if not cc_shift:
                            'No Shift for {} {} {}'.format(cenwave, fpoff, segment)
                            continue
                        with fits.open(old_disp_name) as old_disp:
                            wh_old_disp = np.where((old_disp[1].data['SEGMENT'] == segment) &
                                                   (old_disp[1].data['CENWAVE'] == cenwave) &
                                                   (old_disp[1].data['APERTURE'] == 'PSA'))[0][0]
                            #coeffs_to_update = old_disp[1].data[wh_old_disp]['COEFF']
                            d_to_update = old_disp[1].data[wh_old_disp]['D']

                        with fits.open(new_disp_name, mode='update') as hdu:
                            wh_disp = np.where((hdu[1].data['SEGMENT'] == segment) &
                                        (hdu[1].data['CENWAVE'] == cenwave) &
                                        (hdu[1].data['APERTURE'] == 'PSA'))[0][0]

                            ## Updating the zero-point in the COEFF value, not the D value.
                            #hdu[1].data['COEFF'][wh_disp][0] = coeffs_to_update[0] - (cc_shift*coeffs_to_update[1])
                            hdu[1].data['D'][wh_disp] = d_to_update + cc_shift

                            #print('UPDATING {} {} value={}A difference={}pix {}A'.format(cenwave, segment,
                            #                                                     coeffs_to_update[0] - (cc_shift*coeffs_to_update[1]),
                            #                                                     cc_shift, cc_shift*coeffs_to_update[1]))

                        with fits.open(new_disp_name) as disp:
                            new_d = disp[1].data[wh_disp]['D']
                            print('Updating {}/{}: {} to {} \n Difference of: {}'.format(cenwave, segment, d_to_update, new_d, cc_shift))

# -------------------------------------------------------------------------------
if __name__ == '__main__':

    # These are output files that updating_lamptab uses
    today = datetime.now()
    today_str = today.strftime("%m%d%y_%s")
    

    refdir='/user/cmagness/newcenwaves/lamptab_files/' #cloned new cenwave repo
    #datadir1='/Users/bjames/COS/calibration/G160M_1533/lamptab/data/'
    datadir2='/grp/hst/cos2/c800/data/15484/'
    #corrdir1 = os.path.join(datadir1,"/rextracted_corrtags_v1/") # PERHAPS COMBINE THESE HERE??
    corrdir2 = os.path.join(datadir2,"corrdir/rextracted_corrtags_new/") # PERHAPS COMBINE THESE HERE??
    
    save_file = os.path.join(refdir,'final_FP_shifts_new.txt')

    
    newlampfiles = sorted(glob.glob(os.path.join(corrdir2,'*_x1d.fits'))) #+glob.glob(os.path.join(corrdir2,'*_x1d.fits'))) # adding both lists of files and sorting
    #newlampfiles = sorted(glob.glob(os.path.join(corrdir1, '*_x1d.fits')) + glob.glob(os.path.join(corrdir2, '*_x1d.fits')))  # adding both lists of files and sorting

    #newlampfiles = sorted(glob.glob(os.path.join(corrdir1,'/*_x1d.fits'))) # just 1533
    # name of new lamptab to be made in this code
    new_lamp_name = os.path.join(refdir,"newcenwave_fuv_{}_lamp.fits".format(today_str))
    # LP1 lamptab
    old_lamp_name = os.path.join(refdir,"2018-07-03_v0_lamp.fits")#HERE!!

    # this function creates the new lamp template file --> returns nothing but creates file
    makelamptemplatefile(new_lamp_name, old_lamp_name, newlampfiles)

    # what you want the new dispersion table to be called
    new_disp_name = os.path.join(refdir,'newcenwave_fuv_{}_disp.fits'.format(today_str))
    # interim disptab created by Will
    old_disp_name = '/user/cmagness/newcenwaves/newcenwaves_disp.fits'#HERE

    # finding FP_PIXEL_SHIFT and creating an output txt file
    #os.path.exists(save_file):
    find_fp_pix_shift(new_lamp_name, save_file)

    # Need to update the LAMPTAB before updating the DISPTAB because there are dependencies
    #if os.path.isfile(new_lamp_name):
    update_lamptab(save_file, new_lamp_name)
    #else:
     #   raise ValueError('Exiting. Lamptab file does not exist.')

    # Updating DISPTAB
    # copying the old disptab so the linear dispersion values match when the
    # LP1 lamptab was created.
    if os.path.exists(new_disp_name):
        y_n = input('Do you want to overwrite {}? y/n '.format(new_disp_name))
        if y_n == 'y':
            os.remove(new_disp_name)
            shutil.copy(old_disp_name, new_disp_name)
        else:
            raise ValueError("Exiting. Change DISPTAB name.")
    else:
        shutil.copy(old_disp_name, new_disp_name)

    update_zeropoint(new_lamp_name, old_lamp_name, new_disp_name, old_disp_name)
