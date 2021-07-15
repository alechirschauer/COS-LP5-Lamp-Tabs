import os
import glob
from astropy.io import fits
import numpy as np
import calcos
import shutil
from matplotlib import pyplot as plt


# -------------------------------------------------------------------------------
def get_corrtag_filename(datadirec, filename, segment):

    rootname = filename.split('/')[-1].split('_')[0]

    if segment == 'FUVA':
        aorb = 'a'
    elif segment == 'FUVB':
        aorb = 'b'
    else:
        raise ValueError('segment is neither A or B')

    corrtagfile = os.path.join(datadirec, rootname + '_corrtag_' + aorb + '.fits')

    return corrtagfile


# -------------------------------------------------------------------------------
def get_corrtag_info(datadirec, filename, segment):

    corrtagfile = get_corrtag_filename(datadirec, filename, segment)

    inform = {}
    with fits.open(corrtagfile) as f:
        inform['grating'] = f[0].header['OPT_ELEM']
        inform['cenwave'] = f[0].header['CENWAVE']
        inform['segment'] = f[0].header['SEGMENT']
        inform['rootname'] = f[0].header['ROOTNAME']
        inform['aperture'] = f[0].header['APERTURE']
        inform['fppos'] = f[0].header['FPOFFSET']+3
        inform['exptime'] = f[1].header['EXPTIME']
        inform['xtractab'] = f[0].header['XTRACTAB'].split('$')[-1]  # only want the table name, not the lref$..

        timearr = f[1].data['TIME']
        xcorrarr = f[1].data['XCORR']
        ycorrarr = f[1].data['YCORR']

    
    return inform, xcorrarr, ycorrarr, timearr


# -------------------------------------------------------------------------------
def get_xtraction_boundaries(inform):
    """ Roughly estimate the xtraction boundaries. This isn't the best way to do
        this, but it's easy and a decent estimate
***        Need to double check which file is needed here
    """
    with fits.open(inform['xtractab']) as table:
        index = np.where((table[1].data['APERTURE'] == inform['aperture']) &
                         (table[1].data['CENWAVE'] == inform['cenwave']) &
                         (table[1].data['SEGMENT'] == inform['segment']))
        xtract_data = table[1].data[index]
    lower_bound = xtract_data['B_SPEC'] - (xtract_data['HEIGHT']/2)
    upper_bound = xtract_data['B_SPEC'] + (xtract_data['HEIGHT']/2)

    return lower_bound, upper_bound


# -------------------------------------------------------------------------------
def make_image(xarr, yarr):
    """ This makes the events list into a 2D image array
    """
    # Initialize image size
    new_img = np.zeros((1024, 16384))

    # Add a count for each x,y pair
    for x, y in zip(xarr, yarr):
        x = int(x)
        y = int(y)
        try:
            new_img[y, x] += 1
        except IndexError:
            continue

    return new_img


# -------------------------------------------------------------------------------
def extract_spectrum(inform, x, y):
    """ Extract a spectrum out of the corrtag events
    """
    lower, upper = get_xtraction_boundaries(inform)
    image = make_image(x, y)
    spectrum = np.sum(image[np.int(lower):np.int(upper), :], axis=0)

    return spectrum


# -------------------------------------------------------------------------------
def make_time_blocks_sec(lampflash, inform):
    """ This chops the data (x, y and time) by time so that you can look at the
        spectrum through time
    """

    # opening the lampflash data and getting the LAMP_ON times to make the blocks
    with fits.open(lampflash) as hdu:
        wh_seg = np.where(hdu[1].data['segment'] == inform['segment'])[0]

        on_time = hdu[1].data['LAMP_ON'][wh_seg]

    blockvals = []
    for counter, on in enumerate(on_time):

        if counter == 0:
            # Start at 0 just in case there were counts before the lamp went on
            on = 0
        if counter+1 != len(on_time):
            # if it's not the last on time, the off time is the next on time
            #   it will be non-inclusive in the IF statement that checks.
            off = on_time[counter+1]
        else:
            # the +0.1 is so we can get all of the events that have the exptime
            #   since the IF statemetns are non-inclusive for the "off" time
            off = inform['exptime']+0.1

        blockvals.append([on, off])

    return blockvals

#-------------------------------------------------------------------------------
def correct_xcorr_drift(time, xcorr, blocks, drift):

    # blocks is a list with elements in the format [on, off]
    #  so blocks[0][0] is the first time bin, on time
    #  blocks[0][1] is the first time bin, off time
    #  blocks[-1][1] is the last time bin, off time

    newxcorr = xcorr[np.where((time >= blocks[0][0]) &
                              (time < blocks[0][1]))]

    if time.max() > blocks[-1][1]:
        # sometimes the exptime in the header is wrong
        print('EXPTIME is wrong {}. Changing it to: {}'.format(blocks[-1][1], time.max()))
        blocks[-1][1] = time.max()+0.1

    for d, index in zip(drift, blocks[1:]):
        on = index[0]
        off = index[1]
        wh_time = np.where((time >= on) & (time < off))[0]

        shiftedx = xcorr[wh_time] - d # subtracting the drift!!
        newxcorr = np.append(newxcorr, shiftedx)

    if len(newxcorr) != len(xcorr): # hopefully this shouldn't happen anymore
        import pdb; pdb.set_trace()

    return np.array(newxcorr)
# -------------------------------------------------------------------------------
def write_new_corrtag(datadirec, filename, newxcorrarr, outputdirec, segment):

    corrtagfile = get_corrtag_filename(datadirec, filename, segment)

    outputfilename = corrtagfile.split('/')[-1]

    with fits.open(corrtagfile) as f:
        print(len(f[1].data['XCORR']), len(f[1].data['XFULL']), len(newxcorrarr))
        f[1].data['XCORR'] = newxcorrarr
        f[1].data['XDOPP'] = newxcorrarr
        f[1].data['XFULL'] = newxcorrarr
        f.writeto(os.path.join(outputdirec, outputfilename), clobber=True)

    print('wrote new corrtag {}'.format(os.path.join(outputdirec, outputfilename)))

    return os.path.join(outputdirec, outputfilename)


# -------------------------------------------------------------------------------
def copy_xfull(oldcorrtagfile, outputdir):

    with fits.open(oldcorrtagfile) as f:
        f[1].data['XFULL'] = f[1].data['XCORR']
        f.writeto(os.path.join(outputdir, oldcorrtagfile.split('/')[-1]), clobber=True)

    print('copied corrtag with xfull = xcorr {}'.format(os.path.join(outputdir, oldcorrtagfile.split('/')[-1])))

    return


# -------------------------------------------------------------------------------
def make_new_x1ds(inputfilename, outputdirec):

    calcos.calcos(inputfilename, outdir=outputdirec)

    return


# -------------------------------------------------------------------------------
if __name__ == '__main__':
    
    #datadir='/grp/hst/cos2/c1533/data/15459/corrdir/'
    #datadir='/Users/bjames/COS/calibration/G160M_1533/lamptab/data'
    datadir='/grp/hst/cos2/c800/data/15484/corrdir/'
    
    corrdir = os.path.join(datadir,"wavecorr_on")  #HERE
    outputdir = os.path.join(datadir,"deshifted_corrtags_new")  #HERE
    calcosout = os.path.join(datadir,"rextracted_corrtags_new")  #HERE
    
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    if not os.path.exists(calcosout):
        os.makedirs(calcosout)    
    
    lffiles = glob.glob(os.path.join(corrdir, '*lampflash.fits'))
    print(corrdir)
    for lampflash in lffiles:
        
        # open the lampflash file and grab the data and header
        with fits.open(lampflash) as df:
            lampdata = df[1].data
            lamphead = df[0].header
        
        cenwave = lamphead['cenwave']
        fppos = lamphead['fppos']
            
            # open the lamptab file and grab the data
            # this should be the interim lamptab file
        lamptab = '/user/cmagness/newcenwaves/lamptab_files/2018-07-03_v0_lamp.fits'#HERE
        with fits.open(lamptab) as lt:
            ltdata = lt[1].data
                
        for segment in ['FUVA', 'FUVB']:
            if (cenwave == 800) or (cenwave == 1533):
                if (cenwave == 800) & (segment == 'FUVB'):
                    continue
                
                
                shifts = lampdata[np.where(lampdata['segment'] == segment)]['SHIFT_DISP']
                
                wh_lt = np.where((ltdata['segment'] == segment) &
                                 (ltdata['cenwave'] == cenwave) &
                                 (ltdata['fpoffset'] == fppos-3) )
                fp_pixel_shift = ltdata[wh_lt]['fp_pixel_shift'][0]
                
                subshifts = shifts - fp_pixel_shift
                finalshifts = np.array([x - subshifts[0] for x in subshifts[1:]])
                
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('shifts found for {} {} {} = {}'.format(cenwave, fppos, segment, finalshifts))
                
                # if any shifts are greater than 1 pixel, remove the drift
                
                # if np.any(abs(finalshifts) > 1.0):

                print('correcting xcorr....')
                info, xcorr, ycorr, time = get_corrtag_info(corrdir, lampflash, segment)
                
                # blocks are in the format [time_on, time_off] for a single setting from the lampflash
                blocks = make_time_blocks_sec(lampflash, info)
                print(blocks)
                
                newxcorr = correct_xcorr_drift(time, xcorr, blocks, finalshifts)
                newcorrtagfile = write_new_corrtag(corrdir, lampflash, newxcorr, outputdir, segment)
                
                # else:
                #     print('no correction done')
                #     oldcorrtagfile = get_corrtag_filename(datadir, lampflash, segment)
                #     copy_xfull(oldcorrtagfile, outputdir)
                
    newcorrtags = glob.glob(os.path.join(outputdir, '*corrtag*'))
                
    for newcorrtag in newcorrtags:
        fits.setval(newcorrtag, 'WAVECORR', value='OMIT')
            
    newcorrfilesa = [x for x in newcorrtags if 'corrtag_b' not in x]
                
    for newcorrtaga in newcorrfilesa:
        make_new_x1ds(newcorrtaga, calcosout)
                    
