from astropy.io import fits
import glob
import os
import numpy as np

from matplotlib import pyplot as plt

#-------------------------------------------------------------------------------

def get_shifts(shifts):

    new_shifts = []
    for shift in shifts[1:]:
        new_shifts.append(shift-shifts[0])

    #print('{:5} {:3} {:6} {:10}'.format(cenwave, fp, segment, max_shift ))
    return new_shifts

#-------------------------------------------------------------------------------

def make_figures(lamp_list):

    for lamp in lamp_list:

        outdir = '/Users/ahirschauer/Documents/Year4/07-2021/LAMPTABs/Individual_Flashes_Plots/output/'
        initial_dir = '/Users/ahirschauer/Documents/Year4/07-2021/LAMPTABs/Individual_Flashes_Plots/pre-shift/'
        new_dir = '/Users/ahirschauer/Documents/Year4/07-2021/LAMPTABs/Individual_Flashes_Plots/post-shift/'

        lamp_init = os.path.join(initial_dir, lamp)
        lamp_fin = os.path.join(new_dir, lamp)

        colors = {1: 'red',
                  2: 'blue',
                  3: 'orange',
                  4: 'green'}

        for segment in ['FUVA', 'FUVB']:
            plt.figure(figsize=(10, 5))
            shifts = []
            for lamp_file in [lamp_init, lamp_fin]:

                if 'old' in lamp_file:
                    marker = 'x'
                else:
                    marker = '.'

                with fits.open(lamp_file) as lf: # reading in the lamptab file
                    lamptab = os.path.join("/grp/hst/cos2/LP5_ERA/files_to_use/LP3_G140L_fuv_15July2021_lamp.fits"), lf[0].header['LAMPTAB'].split('$')[-1]

                    print ('-----------------')
                    print (lamptab)
                    cen = lf[0].header['cenwave']
                    fp = lf[0].header['fpoffset']
                    print (cen,fp+3,segment)
                    with fits.open(lamptab) as lt:
                        wh_tab = np.where((lt[1].data['cenwave'] == cen) &
                                          (lt[1].data['fpoffset'] == fp) &
                                          (lt[1].data['segment'] == segment))

                        if len(wh_tab[0]) == 0:
                            print('no {} {} {}'.format(cen, fp, segment))
                            continue
                        fp_pix_shift = lt[1].data['FP_PIXEL_SHIFT'][wh_tab] #finding FP_PIXSHIFT for that lampflash

                    wh_seg = np.where(lf[1].data['segment'] == segment)

                    #shift_disp is the shift between tagflash wavecal and template wavecal in dispersion direction
                    #i.e. the FP_shift + any drift
                    shift_disp = lf[1].data['SHIFT_DISP'][wh_seg]
                    print ('shift_disp:')
                    print (shift_disp)

                    shift_disp = shift_disp - fp_pix_shift
                    # since comparison was made between de-drifted corrtags and lamptemplate corrected for drift,
                    # the resultant value should be the FP shifts. Subtracting FP shifts should give zero
                    print ('shift_disp - FP_pixshift:')
                    print (shift_disp)

                    new_shifts = get_shifts(shift_disp)
                    print ('shifts relative to the first one (i.e. drift)')
                    print(new_shifts)

                    xvals=[2,3,4]
                    plt.plot(xvals,new_shifts, marker=marker, color=colors[fp+3], markersize=15, linestyle="None",
                             label='{} FP{}'.format(lamp_file.split('/')[-2], lf[0].header['FPPOS']))

                    shifts.append(new_shifts)

            plt.title('{} FP{} {}'.format(cen, fp+3, segment))
            plt.xlabel('Exposure after 1st')
            plt.ylabel('Drift Relative to 1st Exposure')
            plt.axhline(0, ls='--', color='black')
            plt.axhline(-0.5, ls=':', color='orchid')
            plt.axhline(0.5, ls=':', color='orchid')

            #plt.legend(bbox_to_anchor=(0.2, 0.98), ncol=4)
            plt.legend(loc='upper left', ncol=4)

            shifts = np.array(shifts)
            if len(shifts) == 0:
                plt.close()
                continue
            if np.max(abs(shifts)) > 1.0:
                plt.ylim(-2.0, 2.0)
#                plt.ylim(-1*(np.max(abs(shifts)))-0.1, (np.max(abs(shifts))+0.1) )
            else:
                plt.ylim(-1.0, 1.0)

            outfile = os.path.join(outdir, '{}_{}_{}_compare.png'.format(cen, fp+3, segment))
            plt.savefig(outfile)
            print('Saved: {}'.format(outfile))
            plt.close()

if __name__ == "__main__":

    outdir = '/Users/ahirschauer/Documents/Year4/07-2021/LAMPTABs/Individual_Flashes_Plots/output/'
    initial_dir = '/Users/ahirschauer/Documents/Year4/07-2021/LAMPTABs/Individual_Flashes_Plots/pre-shift/'
    new_dir = '/Users/ahirschauer/Documents/Year4/07-2021/LAMPTABs/Individual_Flashes_Plots/post-shift/'

    # grabbing the basenames of every lampflash in the newly calibrated directory
    lamp_bases = [os.path.basename(f) for f in glob.glob(os.path.join(new_dir, '*lampflash*'))]
    make_figures(lamp_bases)
