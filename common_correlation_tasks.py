from __future__ import division
import numpy as np
import scipy
import os.path
from astropy.io import fits, ascii

from scipy import integrate
from scipy.stats import pearsonr
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
from astropy.table import Table
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from stsci.convolve import boxcar

import math

# plt.ion()


# -----------------------------------------------------------------------------------------
def downsample_1d(myarr,factor):
    """
    Downsample a 1D array by averaging over *factor* pixels.
    Crops right side if the shape is not a multiple of factor.

    Got this specific function from "Adam Ginsburg's python codes" on agpy

    myarr : numpy array

    factor : how much you want to rebin the array by
    """
    xs = myarr.shape[0]
    crarr = myarr[:xs-(xs % int(factor))]
    dsarr = np.mean(np.concatenate(
                     [[crarr[i::factor] for i in range(factor)]]
                     ), axis=0)

    return dsarr


# -----------------------------------------------------------------------------------------
def linearize(flux, wave, dispersion=None):
    """ Return flux and wave arrays with a linear wavelength scale
        """

    interp_func = interp1d(wave, flux, 1, bounds_error=False, fill_value=0)
    if not dispersion:
        dispersion = np.median((wave - np.roll(wave,1)))
    out_wave = np.arange(wave.min(), wave.max(), dispersion)
    out_flux = interp_func(out_wave)

    return out_flux, out_wave


# -----------------------------------------------------------------------------------------
def calculate_coeff(a, b, shift):
    # for correlation coefficent
    if shift > 0:
        a = a[shift:]
        b = b[:-shift]
    elif shift < 0:
        a = a[:shift]
        b = b[-shift:]
    # else if shift == 0 then we just use a and b as is
    corr_coeff = abs(pearsonr(a, b)[0])

    return corr_coeff


# -----------------------------------------------------------------------------------------
def quad_fit(c, minpix=5, maxpix=5):

    if len(c) == 1:
        return None
    x = np.arange(len(c))
    if np.argmax(c)-minpix > 0:
        x = x[np.argmax(c)-minpix: np.argmax(c)+(maxpix+1)]
        c2 = c[np.argmax(c)-minpix: np.argmax(c)+(maxpix+1)]
    else:
        x = x[0: np.argmax(c)+(maxpix+1)]
        c2 = c[0: np.argmax(c)+(maxpix+1)]
    try:
        quad_fit = np.poly1d(np.polyfit(x, c2, 2))
        new_shift = (-quad_fit[1]/(2*quad_fit[2]))  # zero point -b/2a
    except ValueError:
        import pdb; pdb.set_trace()
    return new_shift


# -----------------------------------------------------------------------------------------
def correlation(a, b, alims=(0,-1), blims=None, dispersion=None, normalize=True,
                direct=True, fft=False, fit_peak=True):
    if blims == None:
        blims=alims

    if len(a) > len(b):
        alims = (0, -1* (1 + len(a) - len(b)))

    if len(b) > len(a):
        blims = (0, -1 * (1 + len(b) - len(a)))

    if normalize:
        # need to normalize this way
        a = (a-a.mean())/a.std()
        b = (b-b.mean())/b.std()

    # redefine the array lengths so the array sizes match
    a = a[alims[0]:alims[1]]
    b = b[blims[0]:blims[1]]

    if direct:
        shift, c, corr_coeff = direct_correlate(a, b, fit_peak)
        if shift == None:
            return None, None
    elif fft:
        shift, c, corr_coeff = fft_correlate(a, b)

    if dispersion:
        shift = shift * dispersion

    return shift, corr_coeff


# -----------------------------------------------------------------------------------------
def fft_correlate(a, b):
    """ Perform FFT correlation between two spectra

        Normalization of the cross-correlation function is computed by dividing
        the cross-correlation by sqrt( sum(a**2) * sum(b**2) ).
        """
    # compute the cross-correlation
    c = (scipy.ifft(scipy.fft(a)*scipy.conj(scipy.fft(b)))).real
    # Normalize the cross-correlation output
    c /= np.sqrt(np.sum(a**2) * np.sum( b**2))
    shift = np.argmax(c)

    if shift > len(a)/2.0:
        shift = shift - (len(a)-1)

    corr_coeff = calculate_coeff(a, b, shift)

    return shift, c, corr_coeff


# -----------------------------------------------------------------------------------------
def direct_correlate(a, b, fit_peak):

    # direct correlation
    c = np.correlate(a, b, mode='full')
    shift = np.argmax(c)
    if np.isnan(c).any():
        return None, None, None
    if fit_peak:
        shift = quad_fit(c)
    if shift == None:
        return None, None, None

    shift = shift - (len(a)-1)

    corr_coeff = calculate_coeff(a, b, int(round(shift)))

    return shift, c, corr_coeff


# -----------------------------------------------------------------------------------------
def cross_correlate(flux_a, flux_b, wave_a=None, wave_b=None, subsample=1):
    """ Cross correlate two spectra in wavelength space
        Flux A is the reference spectra
        """

    dispersion = np.median((wave_a - np.roll(wave_a,1)))
    if dispersion == 0:
        raise ValueError('Dispersion needs to be GT 0')
    # sub sample
    dispersion /= subsample

    low_wave = max(np.nanmin(wave_a), np.nanmin(wave_b))
    high_wave = min(np.nanmax(wave_a), np.nanmax(wave_b))

    index = np.where( (wave_a <= high_wave) & (wave_a >= low_wave) )[0]
    flux_a = flux_a[ index ]
    wave_a = wave_a[ index ]

    index = np.where( (wave_b <= high_wave) & (wave_b >= low_wave) )[0]
    flux_b = flux_b[ index ]
    wave_b = wave_b[ index ]

    flux_a, wave_a = linearize( flux_a, wave_a, dispersion )
    flux_b, wave_b = linearize( flux_b, wave_b, dispersion )

    shift, corr_coeff = correlation( flux_a, flux_b, dispersion=dispersion )

    return shift, corr_coeff


# -----------------------------------------------------------------------------------------
def find_centroid(x, y, weight=1):
    """Calculated the centroid of the input distribution.  Computed as the
        integral of (x * y * weight) / integral of (weight * y).

        Parameters
        ----------
        x : np.ndarray
        x values of distribution
        y : np.ndarray
        y values of distribution
        weight : float, int, np.ndarray, optional
        weights for y values

        Returns
        -------
        centroid : float
        position of the centroid
        error : float
        error estimate of the centroid location

        """

    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    y = y[~np.isinf(x)]
    x = x[~np.isinf(x)]

    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    x = x[~np.isinf(y)]
    y = y[~np.isinf(y)]

    if len(x) != len(y):
        raise ValueError("x and y not equal lengths: {} {}".format(x, y))

    if len(x) == 0:
        return 0, -999

    centroid = integrate.simps(weight*x*y, x) / integrate.simps(weight*y, x)
    error = integrate.simps(weight * y * (x - centroid)**2) / integrate.simps(weight * y)**2

    return centroid, error


# -----------------------------------------------------------------------------------------
def plot_in_loop(ref_wave, ref_flux, cos_wave, cos_flux, shift, cos_centroid=None, stis_centroid=None):
    '''
    This function plots the stis spectrum (or part of spectrum), cos spectrum
    before any manual shift, and cos spectrum after any shift. It also has
    optional parameters to plot the centroids that are calculated as well.

    Parameters
    ----------
    ref_wave : array
        the reference wavelengths that you want to be plotted ; best if this is
        normalized to match COS if STIS spectrum
    ref_flux : array
        the corresponding reference fluxes
    cos_wave : array
        the COS wavelengths to be overplotted on the STIS spectrum ; this needs
        to correspond to approximately the same range as the STIS wavelengths.
    cos_flux : array
        the corresponding COS fluxes
    shift : float
        the value you want cos_wave shifted by. This should be in angstroms
        if you are plotting wavelengths from a COS x1d.
    cos_centroid : float (optional)
        the calculated COS centroid you want plotted
    stis_centroid : float (optional)
        the calculated STIS centroid you want plotted
    '''

    plt.figure()
    plt.plot(ref_wave, ref_flux, color = 'black', label = 'ref_wave vs ref_flux reference')
    plt.plot(cos_wave, cos_flux, label = 'cos_wave vs cos_flux', color='green')
    plt.plot(cos_wave-shift, cos_flux, label = 'cos_wave shifted vs cos_flux', color='blue')

    y_line_arr = np.arange(2)*cos_flux.max()
    if cos_centroid != None:
        plt.plot(np.zeros(len(y_line_arr))+cos_centroid, y_line_arr, color = 'green', ls = '--')
    if stis_centroid != None:
        plt.plot(np.zeros(len(y_line_arr))+stis_centroid, y_line_arr, color = 'black', ls = '--')

    # labels
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Wavelength Bin = {} to {} | Shift: {}'.format(cos_wave.min(), cos_wave.max(), shift))
    plt.legend(loc = 'best', fontsize = 10)
    plt.show()
    # x = raw_input('Press enter to continue \n')
    # plt.close()


# -----------------------------------------------------------------------------------------
def if_emission(targ):

    emission_targets = ['CD-34D7151', 'DR-TAU', 'HD39587', 'HD150798' , 'HD164058',
                        'HD169142', 'HD209458', 'HD40307', 'HD432',  'HD82210', 'HD855112',
                        'HD97658', 'RHO-CNC', 'V-EPS-ERI']
    cos_targ = map_names(targ)
    if cos_targ in emission_targets:
        emission = True
    else:
        emission = False

    return emission


# -----------------------------------------------------------------------------------------
def pull_data(hdu, wmin, wmax, sav_file=False, seg=None):

    blank = np.array([])
    # STIS sav file
    if sav_file:
        index = np.where((hdu.wave_cos >= wmin) &
                         (hdu.wave_cos <= wmax))[0]
        if len(index) > 1:
            return hdu.wave_cos[index], hdu.spec_cos[index], np.ones(len(hdu.spec_cos[index])), None, index[0], index[-1]
        else:
            return blank, blank, blank, None, None, None

    # otherwise treat as a fits file
    for row in hdu[1].data:
        if seg != None:
            index = np.where((row['wavelength'] >= wmin) &
                             (row['wavelength'] <= wmax) &
                             (row['segment'] == seg))[0]
        else:
            index = np.where((row['wavelength'] >= wmin) &
                             (row['wavelength'] <= wmax))[0]
        # Because it probably will only overlap on one segment ; although not true for NUVC for G230L
        if len(index) > 1:
            try:
                return row['wavelength'][index], row['flux'][index], row['dq_wgt'][index], row['segment'], index[0], index[-1]
            except KeyError:
                return row['wavelength'][index], row['flux'][index], np.ones(len(index)), None, index[0], index[-1]

    return blank, blank, blank, None, None, None


# -----------------------------------------------------------------------------------------
def calculate_xcorr(wave, header0, header1, segment, disp):

    # first step is removing the orbital motion(v_helio)
    v_helio = header1['V_HELIO']
    c = 299792. # km/s
    helio_wave = wave / (1-v_helio/c)  # final product from this section
    # second step is taking out the dispersion ; this is in pixel space I think
    wh_setting = np.where((disp['CENWAVE'] == header0['CENWAVE']) &
                           (disp['SEGMENT'] == segment) &
                           (disp['APERTURE'] == 'PSA'))[0]
    a0 = disp[wh_setting]['COEFF'][0][0]
    a1 = disp[wh_setting]['COEFF'][0][1]
    a2 = disp[wh_setting]['COEFF'][0][2]
    dtv03 = disp[wh_setting]['D_TV03'][0]
    d = disp[wh_setting]['D'][0]

    disp_wave = (helio_wave - a0) / a1 - (dtv03 - d)  # final product from this section
    # third step is taking out the wavelength correction (shift1a/b)

    shift_wave = disp_wave + header1['shift1{}'.format(segment[-1].lower())] # final product from this section
    # last step is correcting for the doppler motion of the orbit

    kdopp = (header1['DOPPMAGV'] / c) / a1
    t_start = (header1['EXPSTART']-header1['DOPPZERO']) * 86400 # converting to seconds from days
    t_end = (header1['EXPEND'] - header1['DOPPZERO']) * 86400
    mean_amp = 2 * math.pi / header1['ORBITPER']  # mean amplitude of the orbital period (sine wave)
    xdopp = ( (kdopp * helio_wave * (math.cos(mean_amp*t_end) - math.cos(mean_amp*t_start)) ) / mean_amp) / (t_end - t_start)
    xcorr = shift_wave - xdopp  # final product from this section

    return xcorr


# -----------------------------------------------------------------------------------------
def correlations(dataset, reference, wave_list=None, binsize=None, store_dic=None):

    """This is the main function that cross-correlates specific windows of
    COS with the same windows for Reference.

    Parameters
    ----------
    dataset : str
        name of file to be correlated to
    reference : str
        name of file that will be used as the reference
    wave_list : str, optional
        the location of the ascii file that has the wavelength min/maxs
    binsize : int, optional
        the size of the steps to use when stepping through
    store_dic : dictionary, optional
        an empty dictionary to use to save parameters ; this has been changed
        to use an ascii table instead

    Returns
    --------
    out_table : Table
        All of the parameters we would like saved into a file
    """
    num_windows = 0
    quality = None

    obs = fits.open(dataset)
    disptab_name = obs[0].header['DISPTAB']
    if 'lref' in disptab_name:
        disptab_name = os.path.join('/grp/hst/cdbs/lref', disptab_name.split('$')[-1])
    disptab = fits.getdata(disptab_name)

    if reference.endswith('.sav'):
        ref = readsav(reference)
        ref_head0 = {}
        ref_head1 = {}
        sav_file = True
        if not store_dic:
            make_dictionary('stis')
    else:
        ref = fits.open(reference)
        ref_head0 = ref[0].header
        ref_head1 = ref[1].header
        sav_file = False
        if not store_dic:
            make_dictionary('cos')

    if binsize != None:
        windows = []
        for wave_data in ref[1].data['wavelength']:
            for start in np.arange(0, len(wave_data), binsize):
                if start + binsize > len(wave_data):
                    continue
                windows.append((wave_data[start], wave_data[start+binsize]))
    elif wave_list != None:
        window_data = ascii.read(wave_list)
        windows = [(float(row[0]), float(row[1])) for row in window_data]
        if len(row) > 2:
            quality = [float(row[2]) for row in window_data]
    else:
        raise ValueError('You have to decide on a binsize or window file')

    # loop over datasets performing correlations
    num_windows = 0
    correlation_data = []
    for ind, (wmin, wmax) in enumerate(windows):
        if quality != None:
            qual = quality[ind]
        else:
            qual = -1
        if reject_windows(wmin, obs[0].header['rootname']):
            # the rootname + window combination is bad
            continue

        if (wmin >= 1210 and wmin <= 1220) or (wmin >= 1300 and wmin <= 1307) \
            or (wmin >= 1197 and wmin <= 1202):
            continue #airglow
        for segment in obs[1].data['SEGMENT']:
            obs_wave, obs_flux, obs_dq_wgt, obs_segment, obs_xmin, obs_xmax = pull_data(obs, wmin, wmax, seg=segment)
            if not len(obs_wave):
                continue
            ref_wave, ref_flux, ref_dq_wgt, ref_segment, ref_xmin, ref_xmax = pull_data(ref, wmin, wmax, sav_file=sav_file)
            if not len(ref_wave):
                continue

            # -- trim reference spectrum
            wh_overlap = np.where( (ref_wave >= obs_wave.min()) & (ref_wave <= obs_wave.max()))[0]
            ref_flux = ref_flux[wh_overlap]
            ref_wave = ref_wave[wh_overlap]
            ref_dq_wgt = ref_dq_wgt[wh_overlap]
            if (obs_dq_wgt==0).any() or ((ref_dq_wgt==0).any()):
                # If there are any windows that overlap with a DQ weight of zero,
                # then throw away those windows
                continue

            interp_func = interp1d(obs_wave, obs_flux, 1)
            interp_flux = interp_func(ref_wave)

            ## info for Justin
            obs_counts = interp_flux.sum()
            obs_std = interp_flux.std()
            ref_counts = ref_flux.sum()
            ref_std = ref_flux.std()

            # # attemping to do a monte carlo thing to figure out how good our window is.
            # shift_dist = []
            # ref_mu = np.mean(ref_flux)
            # ref_std = np.std(ref_flux)
            # obs_mu = np.mean(obs_flux)
            # obs_std = np.std(obs_flux)
            # for i in np.arange(5000):
                 ##randomizing
            #    ref_rand_flux = ref_flux + np.random.normal(ref_mu, ref_std, len(ref_flux))
            #    obs_rand_flux = obs_flux + np.random.normal(obs_mu, obs_std, len(obs_flux))
                ## smoothing
            if len(ref_flux) > 3 and len(obs_flux) > 3:
                smooth_ref_flux = boxcar(ref_flux, (3,) )
                smooth_interp_flux = boxcar(obs_flux, (3,) )
            else:
                smooth_ref_flux = ref_flux
                smooth_interp_flux = obs_flux

            if len(np.where(smooth_ref_flux)[0]) / len(smooth_ref_flux) < .9:
                continue

            if np.isnan(smooth_ref_flux).any() or np.isnan(smooth_interp_flux).any():
                continue

            try:
                shift, corr_coeff = cross_correlate(smooth_ref_flux, smooth_interp_flux, wave_a=ref_wave, wave_b=ref_wave)
            except ValueError as e:
                print(e)
                continue
            if shift == None:
                continue
            shift *= -1

            #-- find pixel shift by dividing the shift by median dispersion
            pix_shift = shift / np.median((obs_wave - np.roll(obs_wave, 1)))
            #if quality != None and corr_coeff > 0.8 and qual == 3: #and abs(pix_shift) > 7:
            #plot_in_loop(ref_wave, ref_flux, ref_wave, interp_flux, shift)

            if np.isnan(corr_coeff):
                continue
            # if abs(pix_shift) >= 30:
            #    #plot_in_loop(ref_wave, ref_flux, ref_wave, interp_flux, shift)
            #    continue

            # --- Centroiding things
            # to make them easier to compare, I normalize the STIS flux level to the cos flux level
            normalize = np.float64(np.mean(ref_flux) / np.mean(obs_flux))
            normalized_ref = ref_flux / normalize
            # total and check
            if np.isnan(normalized_ref).any():
                print("can't normalize")
                normalize = 1
                normalized_ref = ref_flux  # Sometimes the divide by zero turns these all nans, so it's best to leave it unnormalized

            # finding the centroids
            # if cflux or normalized_ref are used after finding the centroids,
            #    either need to switch back or change this to a temp variable
            emission = if_emission(map_names(obs[0].header['TARGNAME']))
            if not emission:
                cflux = abs(1/obs_flux)
                normalized_ref = abs(1/normalized_ref)

            ## WAVELENGTHS ##
            cos_centroid, cos_error = find_centroid(obs_wave-shift, obs_flux)
            cos_centroid_b, cos_error_b = find_centroid(obs_wave, obs_flux)
            ref_centroid, ref_error = find_centroid(ref_wave, normalized_ref)
            ## PIXELS ##
            pix = np.arange(len(obs_wave)) + obs_xmin
            xfull_cen_before, xfull_error_before = find_centroid(pix, obs_flux) # cos center in pixels before shift
            xfull_cen_after, xfull_error_after = find_centroid(pix-pix_shift, obs_flux)
            if ref_cenwave == 1327:
                plot_in_loop(ref_wave, ref_flux, ref_wave, interp_flux, shift, cos_centroid=cos_centroid, stis_centroid=ref_centroid)

            xcorr_cen_after = calculate_xcorr(cos_centroid, obs[0].header, obs[1].header, obs_segment, disptab)
            xcorr_cen_before = calculate_xcorr(cos_centroid_b, obs[0].header, obs[1].header, obs_segment, disptab)
            if ref_head0.get('INSTRUME', None):
                xcorr_reference = calculate_xcorr(ref_centroid, ref_head0, ref_head1, ref_segment, disptab)
            else:
                xcorr_reference = None

            # These are edge cases it seems ; not showing up anymore
            if np.isnan(cos_centroid) or np.isnan(cos_centroid_b) or np.isnan(ref_centroid):
                print('Nans', cos_centroid, cos_centroid_b, ref_centroid, cos_error, cos_error_b, ref_error)
            # -------------
            correlation_data.append((reference,
                                     map_names(obs[0].header['targname']),
                                     wmin,  # ref_wave.min(),
                                     wmax,  # ref_wave.max(),
                                     ref_xmin,
                                     ref_xmax,
                                     ref_segment,
                                     ref_head0.get('proposid', None),
                                     ref_head0.get('cenwave', None),
                                     ref_head0.get('fppos', None),
                                     ref_head0.get('life_adj', None),
                                     ref_centroid,
                                     xcorr_reference,
                                     # ref_counts,
                                     # ref_std,
                                     dataset,
                                     obs_wave.min(),
                                     obs_wave.max(),
                                     obs_xmin,
                                     obs_xmax,
                                     obs_segment,
                                     obs[0].header['proposid'],
                                     obs[0].header['cenwave'],
                                     obs[0].header['fppos'],
                                     obs[0].header['life_adj'],
                                     cos_centroid_b,
                                     cos_centroid,
                                     xcorr_cen_before,
                                     xcorr_cen_after,
                                     xfull_cen_before,
                                     xfull_cen_after,
                                     # obs_counts,
                                     # obs_std,
                                     shift,
                                     pix_shift,
                                     corr_coeff,
                                     qual,
                                     # np.std(shift_dist),
                                     # np.std(pix_dist),
                                     # obs[1].header['pa_aper'],
                                     # obs[1].header['shift1a'],
                                     # obs[1].header['shift1b'],
                                     # obs[1].header['shift2a'],
                                     # obs[1].header['shift2b'],
                                     obs[1].header['v_helio'],
                                     # obs[1].header['sp_loc_a'],
                                     # obs[1].header['sp_loc_b'],
                                     # obs[1].header['sp_off_a'],
                                     # obs[1].header['sp_off_b'],
                                     # obs[1].header['x_offset'],
                                     # obs[1].header['orientat'],
                                     # obs[1].header['sunangle'],
                                     # obs[1].header['expstart'],
                                     # obs[1].header['exptime'],
                                     # obs[1].header['numflash'],
                                     # obs[1].header['nevents']
                                     normalize
                                     ))

    names = ('reference',
             'targname',
             'ref_wmin',
             'ref_wmax',
             'ref_xmin',
             'ref_xmax',
             'ref_segment',
             'ref_pid',
             'ref_cenwave',
             'ref_fppos',
             'ref_life_adj',
             'ref_centroid',
             'ref_centroid_xcorr',
             # 'ref_counts',
             # 'ref_std',
             'observation',
             'obs_wmin',
             'obs_wmax',
             'obs_xmin',
             'obs_xmax',
             'obs_segment',
             'obs_pid',
             'obs_cenwave',
             'obs_fppos',
             'obs_life_adj',
             'obs_centroid_before',
             'obs_centroid_after',
             'obs_centroid_before_xcorr',
             'obs_centroid_after_xcorr',
             'obs_centroid_before_xfull',
             'obs_centroid_after_xfull',
             # 'obs_counts',
             # 'obs_std',
             'wave_shift',
             'pix_shift',
             'corr_coeff',
             'quality',
             # 'carlo_wave',
             # 'carlo_pix',
             # 'pa_aper',
             # 'shift1a',
             # 'shift1b',
             # 'shift2a',
             # 'shift2b',
             'v_helio',
             # 'sp_loc_a',
             # 'sp_loc_b',
             # 'sp_off_a',
             # 'sp_off_b',
             # 'x_offset',
             # 'orientat',
             # 'sunangle',
             # 'expstart',
             # 'exptime',
             # 'numflash',
             # 'nevents'
             'normalization'
             )

    if not len(correlation_data):
        print(wave_list, 'no data for target', len(correlation_data))
        return Table(names=names)

    out_table = Table(rows=correlation_data, names=names)
    return out_table


# -----------------------------------------------------------------------------------------
def file_making(files, targ_dir, store_dic, lp):

    for file in files:
        if 'x1d.fits' in file and file.startswith('l'):
            filename = os.path.join(targ_dir, file)
            with fits.open(filename) as x1d:
                if x1d[0].header['LIFE_ADJ'] != lp:
                    continue
                if x1d[0].header['INSTRUME'] != 'COS':
                    continue
                if x1d[0].header['POSTARG1'] != 0 or x1d[0].header['POSTARG2'] != 0:
                    continue
                if x1d[0].header['OPT_ELEM'] == 'G140L':
                    continue
                if x1d[0].header['CENWAVE'] == 1055 or x1d[0].header['CENWAVE'] == 1096:
                   continue

                store_dic['filename'].append(filename)
                store_dic['rootname'].append(x1d[0].header['ROOTNAME'])
                store_dic['lp'].append(x1d[0].header['LIFE_ADJ'])
                store_dic['grating'].append(x1d[0].header['OPT_ELEM'])
                store_dic['cenwave'].append(x1d[0].header['CENWAVE'])
                store_dic['fppos'].append(x1d[0].header['FPPOS'])
                store_dic['segment'].append(x1d[0].header['SEGMENT'])
                store_dic['exptime'].append(x1d[1].header['EXPTIME'])
                store_dic['expstart'].append(x1d[1].header['EXPSTART'])
                store_dic['targname'].append(x1d[0].header['TARGNAME'])
                store_dic['proposid'].append(x1d[0].header['PROPOSID'])

    return store_dic


# -----------------------------------------------------------------------------------------
def create_filelist(main_dir, out_file, version, lp):
    store_dic = (
                 ('filename', []),
                 ('rootname', []),
                 ('lp', []),
                 ('grating', []),
                 ('cenwave', []),
                 ('fppos', []),
                 ('segment', []),
                 ('exptime', []),
                 ('expstart', []),
                 ('targname', []),
                 ('proposid', [])
                                  )
    store_dic = OrderedDict(store_dic)  # because we want to keep this order
    for targ_dir, dirs, files in os.walk(main_dir):
        if targ_dir == main_dir:
            continue
        if 'do_not_use' in targ_dir or 'single_cenwave_only' in targ_dir:
            continue
        if 'calibrated_v' in version and version in targ_dir:
            # ersion should have the actual version number in it i.e. version=calibrated_v12
            store_dic = file_making(files, targ_dir, store_dic, lp)
        elif version == 'calibrated' and 'calibrated' not in targ_dir:
            store_dic = file_making(files, targ_dir, store_dic, lp)

    list_names = store_dic.keys()
    data = store_dic.values()
    ascii.write(data, out_file, names=list_names)


# -----------------------------------------------------------------------------------------
def make_dictionary(cos_or_stis):
    """Creates an ordered dictionary for all of the important info to be saved.

    Parameters
    ----------
    cos_or_stis : str
        this is either 'cos' or 'stis' referring to if the reference is a cos
        spectrum or a stis spectrum respectively.

    Outputs
    -------
    store_dic : dic
        An ordered dictionary with all important information
    """
    # things to save
    store_dic = (
                ('TARGNAME', []),  # TARGNAME
                ('COS FILE', []),  # path to COS file
                ('COS ROOTNAME', []),  # ROOTNAME
                ('COS ID', []),  # PROPOSID
                ('LP', []),  # LIFE_ADJ
                ('COS GRATING', []),  # OPT_ELEM
                ('COS CENWAVE', []),  # CENWAVE
                ('FPPOS', []),  # FPPOS
                ('CALCOS VER', []),  # CAL_VER
                # ('SP_LOC_A', []),  # SP_LOC_A
                # ('SP_LOC_B', []),  # SP_LOC_B
                ('V_HELIO', []),  # V_HELIO
                ('MJD', []),  # EXPSTART
                ('XFULL MIN', []),  # Pixel window min
                ('XFULL MAX', []),  # Pixel window max
                ('WAVE MIN', []),  # Wavelength window min
                ('WAVE MAX', []),  # Wavelength window max
                ('SHIFT (WAVE)', []),  # From cross-correlation
                ('SHIFT (PIX Calculated)', []),  # Calculcated from Shift Wave
                ('BEFORE WAVE CEN', []),  # centroid of original cwave
                ('AFTER WAVE CEN', []),  # centroid of cwave+shift(wave)
                ('REF CENTER', []),  # centroid of swave
                ('RFLUX/CFLUX', []),  # Normalization factor
                ('CORR_COEFF', []),  # Correlation Coefficent of cross correlation
                ('SEGMENT', [])
                                 )
    store_dic = OrderedDict(store_dic)  # because we want to keep this order
    # If we are correlating COS to STIS spectrums, we want different values
    if cos_or_stis == 'stis':
        store_dic['STIS GRATING'] = []  # FROM SAV FILE
        store_dic['BEFORE PIX CEN'] = []  # COS Pixel center original
        store_dic['AFTER PIX CEN'] = []  # COS pixel center after shift(pixels)
    elif cos_or_stis == 'cos':
        store_dic['REFERENCE'] = []  # Reference ROOTNAME #before prog ID after COS ROOTNAME
        store_dic['REF FPPOS'] = []  # Reference CENWAVE
        store_dic['REF CENWAVE'] = []  # Reference file cenwaves
        store_dic['REF FILE'] = []  # path to reference file

    return store_dic


# -----------------------------------------------------------------------------------------
def map_names(cos_targ):
    # These targets have different names in STIS vs COS exposures
    # COS on left STIS on right
    # for differently named targets to be mapped to one name
    names_dic = {'AV456': 'AZV456',
                 'AV75': 'AZV75',
                 'HD-204827': 'HD204827',
                 'MRK-509': 'MARK509',
                 'MARK-509': 'MARK509',
                 'NGC-3516': 'NGC3516',
                 'NGC-3783': 'NGC3783',
                 'NGC-4395': 'NGC4395',
                 'NGC5139-UIT-1': 'NGC5139',
                 'NGC5272-ZNG1': 'VZ1128',
                 'NGC-5548': 'NGC5548',
                 'PKS0454-22': 'Q0454-2203',
                 'SK-155': 'SK155',
                 'V-DR-TAU': 'DR-TAU',
                 'V-TW-HYA': 'CD-34D7151',
                 'HD-169142': 'HD169142',
                 'NGC-1705': 'NGC1705-1',
                 'NGC-7469': 'NGC7469',
                 'HD-209458': 'HD209458',
                 # after this point these are two COS targets with the same name
                 'QSO-B1126-041': 'PG1126-041',
                 'IO-AND': 'QSO0045+3926',
                 'IRAS-F04250-5718': 'RBS542',
                 '-16-CYG-A': '16-CYG-A',
                 '-16-CYG-B': '16-CYG-B',
                 '-RHO-CNC': 'RHO-CNC'}
    if cos_targ in names_dic.keys():
        cos_targ = names_dic[cos_targ]

    return cos_targ


# -----------------------------------------------------------------------------------------
def reject_windows(wmin, rootname):
    # these are the minimum wavelengths for the windows we want to reject for these rootnames
    bad_wind_dic = {  # COS- COS rejections
                    'lb1o11m5q': [1511.74],  # LBQS-1435-0134
                    'lb5h06hsq': [1457.1, 1476.83],  # LBQS-0107-0235
                    'lb5h05e4q': [1457.1, 1476.83],  # LBQS-0107-0235
                    'lb5h05ejq': [1476.83],  # LBQS-0107-0235
                    'lb5h06hzq': [1476.83],  # LBQS-0107-0235
                    'lb1o14ihq': [1531.5],  # PG-1338+416
                    'lb1o14i7q': [1646.5],  # PG-1338+416
                    'lb1o10goq': [1652.52],  # LGQS-1435-0134
                    'lcbx02idq': [1483.5],  # PG1126-041
                    'lcbx03c8q': [1483.5],  # PG1126-041
                    'lccl13xpq': [1607.06, 1714.12],  # Q1354+195
                    'lccl13xfq': [1607.06, 1669.78, 1714.12],  # Q1354+195
                    'lccl13kxq': [1669.78],  # Q1354+195
                    'lccl12tzq': [1669.78],  # Q1354+195
                    'lccl12duq': [1669.78],  # Q1354+195
                    # COS-STIS rejections
                    'lbgl01qnq': [1220.21, 1369],  # PG1116+215
                    'labp50ajq': [1452],  # NGC330-B37
                    'labp73guq': [1452],  # NGC330-B37
                    'lcgh01h8q': [1346.5],  #AZV75
                    'lb3s02wtq': [1740.11],  # HD204827
                    'lcrs51e7q': [1194.0],  # AZV75
                    'lcrs51dpq': [1194.0],  # AZV75
                    'lcgh01h4q': [1194.0],  # AZV75
                    'lbxk51kzq': [1194.0],  # AZV75
                    'lcgh01h8q': [1194.0],  # AZV75
                    'lbrn05n7q': [1740.11],  # HD-204827
                    'lb3s04ohq': [1740.11, 1745.6],  # HD-204827
                    'lbrn03rxq': [1740.11],  # HD-204827
                    'lbgl01qpq': [1369.0]  # PG1116+215
                                        }

    break_loop = False
    if rootname in bad_wind_dic.keys():
        for window_min in bad_wind_dic[rootname]:
            if wmin == window_min:
                print(rootname, wmin, 'REJECTED')
                break_loop = True
    return break_loop
