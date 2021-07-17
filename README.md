# COS-LP5-Lamp-Tabs

17 July 2021
Alec S. Hirschauer

These files were used to create the COS LAMPTAB _lamp.fits and _disp.fits files for the move to LP5 (G130M, cenwaves 1291, 1300, 1309, 1318, and 1327), plus G140L/c800 at LP3.

The .ipynb Jupyter Notebooks which contain the approriate process to make these files are dated in their filenames with "15July2021".

The older dated Jupyter Notebooks are earlier versions which were scrapped in favor of the more recent versions.

The testing Notebook creates plots comparing the Flux Intensities of the newer _lamp.fits files against the previous official file(s).

The .py Python routines are taken from Camellia's grit repository, though had to be modified slightly to suit our needs (specifying different LPs and cenwaves).

I believe all except "common_correlation_tasks.py" were called at some point in the final LAMPTAB creation Notebook.

When LAMPTAB file(s) for LP6 need to be made, these Notebooks and called Python routines will hopefully be helpful and useful.

Retrieve the relevant data from MAST, set your own local working directories, and make sure Camellia's routines cover the appropriate LPs and cenwaves.

If you have trouble, please get in touch with me (ahirschauer@stsci.edu), and/or Elaine Frazer and Rachel Plesha, without whom I wouldn't have been able to finish these tasks.

Good luck!  This isn't a difficult procedure, but if you're new to COS and/or Python (like I am/was), it can be challenging.
