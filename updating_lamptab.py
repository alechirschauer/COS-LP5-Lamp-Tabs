from astropy.io import fits, ascii
from astropy.table import Table, vstack

import numpy as np
import calcos
import glob
import os


# ----------------------------------
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

        elif cenwave == 800:
            segmentA = data['segment'][0]
            correctcolumnA = np.where(
                (olddata['segment'] == segmentA)
                & (olddata['cenwave'] == cenwave)
                & (olddata['opt_elem'] == opt_elem)
                & (olddata['fpoffset'] == fpoffset))

            olddata['intensity'][correctcolumnA] = np.round(data['net'][0] * exptime)

        elif cenwave == 1105:
            segmentA = data['segment'][0]

            correctcolumnA = np.where(
                (olddata['segment'] == segmentA)
                & (olddata['cenwave'] == cenwave)
                & (olddata['opt_elem'] == opt_elem)
                & (olddata['fpoffset'] == fpoffset))

            olddata['intensity'][correctcolumnA] = np.round(data['net'][0] * exptime)

        elif cenwave == 1280:  # don't replace segment B of 1280

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
    old.writeto(newfilename, clobber=True)

    old.close()
