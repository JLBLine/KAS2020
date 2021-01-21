import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from lmfit.models import PowerLawModel, GaussianModel, VoigtModel, SkewedGaussianModel, SkewedVoigtModel
from scipy.signal import savgol_filter

##Setup a dictionary containing known emission lines
##We'll use this in the code and as an argument
emiss_line_dict = {}
emiss_line_dict['Ly_alpha'] = 1216
emiss_line_dict['NV'] = 1240
emiss_line_dict['SiIV'] = 1397
emiss_line_dict['CIV'] = 1549
emiss_line_dict['HeII'] = 1640

def get_coords_from_header(header):
    """Takes a spectral cube FITS header and returns RA, Dec, and
    Wavelength arrays"""

    ##Grab the world coord system
    wcs = WCS(header)

    ##Make an array containing all the pixel values in the x axis, using
    ##the NAXIS1 header value
    ra_pix = np.arange(int(header['NAXIS1']))
    ##Only use y z = 0,0 to return just dec
    ##Anytime we don't care about an output, we set it equal to '_'
    ras, _, _ = wcs.all_pix2world(ra_pix,0,0,0)

    dec_pix = np.arange(int(header['NAXIS2']))
    ##Only use x, z = 0,0 to return just dec
    _, decs, _ = wcs.all_pix2world(0,dec_pix,0,0)

    wave_pix = np.arange(int(header['NAXIS3']))
    ##Only use x, y = 0,0 to return just wavelengths
    _, _, wavelengths = wcs.all_pix2world(0,0,wave_pix,0)

    return ras, decs, wavelengths

def plot_spectra(wavelengths, spectra, smoothed=False):
    """Plot a spectra as a function of wavelength on a 1D plot """

    ##Plot em up
    fig, ax = plt.subplots(1,1,figsize=(12,5))

    ax.plot(wavelengths, spectra, 'gray', label='FITS data',alpha=0.7)

    if type(smoothed) == np.ndarray:
        ax.plot(wavelengths, smoothed, 'k', label='Smoothed')

    for colour,key in enumerate(emiss_line_dict.keys()):
        ax.axvline(emiss_line_dict[key],color="C{:d}".format(colour),linestyle='--',label=key)

    ax.legend()
    ax.set_ylabel('$F_{\lambda}$ ($\mathrm{erg}\,\mathrm{s}^{-1} \mathrm{cm}^{-2}$ Å$^{-1}$)')
    ax.set_xlabel('Wavelength (Å)')

    fig.savefig('input_spectra.png',bbox_inches='tight')

def spectra_subset(wavelengths, spectra, width=250, line_cent=False,
                   lower_wave=False, upper_wave=False):
    """Takes the given spectra and wavelengths (in
    angstrom) and crops. If line_cent is provided, crop about line_cent
    to the given width (defaults to 2000 Angstrom). Alternatively,
    can manually specifiy a lower wavelength bounary (lower_wave)
    and upper wavelength bounary (upper_wave). lower_wave and upper_wave
    will overwrite any limits set via line_cent and width."""

    ##Exit if reasonable combo of arguements not given
    if not line_cent and not lower_wave and not upper_wave:
        sys.exit("ERROR in spectra_subset - if line_cent is not set, both lower_wave and upper_wave must be set. Exiting.")

    if line_cent:
        lower_crop = line_cent - width / 2
        upper_crop = line_cent + width / 2

    if lower_wave:
        lower_crop = lower_wave

    if upper_wave:
        upper_crop = upper_wave

    ##The where function returns an array of indexes where
    ##the boolean logic is true
    indexes = np.where((wavelengths >= lower_crop) & (wavelengths <= upper_crop))
    ##Use our result from where to crop the spectra and wavelengths
    return spectra[indexes], wavelengths[indexes]

def do_lmfit(lm_model, trim_wavelengths, trim_spectra, line_cent):
    """Takes an lmfit.models instance (lm_model) and fits a spectra
    as described by trim_spectra, trim_wavelenghts."""

    ##You can combine lmfit models just by adding them like this:
    emiss_and_power_model = lm_model(prefix='emission_') + PowerLawModel(prefix='power_')

    ##You create the params in the same way as you would a single model
    emiss_and_power_params = emiss_and_power_model.make_params()

    ##Start the fitting at the line centre that we want
    emiss_and_power_params['emission_center'].set(value=line_cent)

    ##Do the fit
    fit = emiss_and_power_model.fit(trim_spectra, emiss_and_power_params, x=trim_wavelengths)

    return fit

def do_fit_plot(wavelengths, spectra, trim_wavelengths, trim_spectra, fit):
    """Plots the given input spectra (wavelengths, spectra) the subset of data
    that was used for fitting (trim_wavelenghts, trim_spectra), and the fit
    result out of lmfit (fit)"""

    ##Plot em up
    fig, axs = plt.subplots(1,2,figsize=(12,4))

    axs[0].plot(rest_wavelengths, spectra, 'gray', label='Input spectra',lw=2.0, alpha=0.5)

    for ax in axs:
        ax.plot(trim_wavelengths, trim_spectra, 'k', label='Fitted spectra',lw=2.0)
        ax.plot(trim_wavelengths,fit.best_fit,lw=2.0, label='Fit result')

        ax.set_xlabel('Rest wavelength (Å)')
        ax.legend()

    axs[0].set_ylabel('$F_{\lambda}$ ($\mathrm{erg}\,\mathrm{s}^{-1} \mathrm{cm}^{-2}$ Å$^{-1}$)')

    fig.savefig('fit_results.png',bbox_inches='tight')

if __name__ == '__main__':
    import argparse
    import sys

    ##This sets up a parser, which can read inputs given at the command line
    parser = argparse.ArgumentParser(description='A script to read in a spectral \
             AGN data and fit a given emission line at a given x,y pixel \
             location ')

    ##This is my personal preference, having 'optional' arguments, and then
    ##setting some to required. This is techincally bad form - argparse defaults
    ##to lising any arg with -- as optional in the --help

    ##You'll also notice I've including two names for some arguments. You can
    ##put a shortcut in front of some arguments to have another name for the
    ##argument, so here -f and --fitsfile are equivalent

    parser.add_argument('-f', '--fitsfile', required=True,
        help='REQUIRED: 3D FITS file with AGN data')

    parser.add_argument('-x','--x_pix', type=int, required=True,
        help='REQUIRED: The x coord of the data to fit (pixel coords)')

    parser.add_argument('-y', '--y_pix', type=int, required=True,
        help='REQUIRED: The x coord of the data to fit (pixel coords)')

    parser.add_argument('--redshift', type=float, default=2.73763,
        help='Redshift of the AGN - defaults to 2.73763')

    parser.add_argument('--smooth_data', action='store_true',
        help='If added, apply a SavGol filter to the spectra')

    parser.add_argument('--savgol_wl', type=int, default=21,
        help='Window length in pixels to use in SavGol smooting. \
              Note this parameter must be an odd integer. Default=21')
    parser.add_argument('--savgol_poly', type=int, default=1,
        help='Order of polynomial used in SavGol smooting.')

    parser.add_argument('--emission_line', default='CIV',
        help="Try fitting this known line. Options are: (Angstroms).: \
              'Ly_alpha' (1216), \
              'NV' (1240), \
              'SiIV' (1397), \
              'CIV' (1549), \
              'HeII' (1640). \
              Defaults to 'CIV' ")

    parser.add_argument('--fitting_model', default='SkewedVoigtModel',
        help="Which lmfit model to use when fitting emission. Options \
             are: GaussianModel, VoigtModel, SkewedGaussianModel, \
             SkewedVoigtModel. Defaults to SkewedVoigtModel")

    parser.add_argument('--fit_width', default=250, type=int,
        help="Width of data to fit in Angstrom")

    parser.add_argument('--lower_wave', default=False, type=int,
        help="Instead of fitting a specific width of spectra with --fit_width, \
             specify a lower boundary (in Angstrom) to fit to")

    parser.add_argument('--upper_wave', default=False, type=int,
        help="Instead of fitting a specific width of spectra with --fit_width, \
             specify an upper boundary (in Angstrom) to fit to")

    parser.add_argument('--no_plots', action='store_true',
        help="If added, do not create any plots, just report fit results")

    ##This grabs all of the arguments out of the parser. Now all the arguments
    ##are attributes of args, which we can access and use
    args = parser.parse_args()

    ##This is only going to work on FITS data that has the same structure as
    ##1009_629_2006A_individual_cubes_3D.fits
    with fits.open(args.fitsfile) as hdu:
    ##Get the data from the 2nd hdu entry
        data_cube = hdu[1].data
        fits_header = hdu[1].header

    ##Check whether our input arguments make sense, and exit with error message
    ##if not
    if args.x_pix < 0 or args.x_pix >= fits_header['NAXIS1']:
        sys.exit("ERROR: --x_pix={:d} is outside the range of the FITS NAXIS1={:d}".format(args.x_pix,fits_header['NAXIS1']))

    if args.y_pix < 0 or args.y_pix >= fits_header['NAXIS2']:
        sys.exit("ERROR: --y_pix={:d} is outside the range of the FITS NAXIS2={:d}".format(args.y_pix,fits_header['NAXIS2']))

    if args.fitting_model not in ["GaussianModel", "VoigtModel", "SkewedGaussianModel", "SkewedVoigtModel"]:
        print("Model specified --fitting_model={:s} is not supported. Defaulting to GaussianModel".format(args.fitting_model))
        args.fitting_model = "GaussianModel"

    if args.emission_line not in ['Ly_alpha', 'NV', 'SiIV', 'CIV', 'HeII']:
        print("Line specified in --emission_line={:s} is not supported. Defaulting to CIV".format(args.emission_line))
        args.emission_line = "CIV"

    ##Grab some relevant coords from the header
    ras, decs, wavelengths = get_coords_from_header(fits_header)

    ##Remove redshifting on the spectra
    rest_wavelengths = wavelengths / (args.redshift + 1)

    ##Cut to the specific pixel
    spectra = data_cube[:,args.y_pix,args.x_pix]

    ##Do the smoothing if asked for it
    if args.smooth_data:
        smoothed_spectra = savgol_filter(spectra, args.savgol_wl, args.savgol_poly)
    else:
        ##If no smoothing, set this to False for use in future functions
        smoothed_spectra = False

    ##Plot input data if requested
    if args.no_plots:
        pass
    else:
        plot_spectra(rest_wavelengths, spectra, smoothed=smoothed_spectra)

    ##Set lm_model to the requested function form for the emission peak
    if args.fitting_model == "GaussianModel":
        lm_model = GaussianModel
    elif args.fitting_model == "VoigtModel":
        lm_model = VoigtModel
    elif args.fitting_model == "SkewedGaussianModel":
        lm_model = SkewedGaussianModel
    elif args.fitting_model == "SkewedVoigtModel":
        lm_model = SkewedVoigtModel

    ##Set the line_cent using the emiss_line_dict:
    if args.emission_line == 'Ly_alpha':
        line_cent = emiss_line_dict['Ly_alpha']
    elif args.emission_line == 'NV':
        line_cent = emiss_line_dict['NV']
    elif args.emission_line == 'SiIV':
        line_cent = emiss_line_dict['SiIV']
    elif args.emission_line == 'CIV':
        line_cent = emiss_line_dict['CIV']
    elif args.emission_line == 'HeII':
        line_cent = emiss_line_dict['HeII']

    ##Specify which data to fit, based on whether we are smoothing or not
    if args.smooth_data:
        spectra_to_fit = smoothed_spectra
    else:
        spectra_to_fit = spectra

    ##Trim the spectra as required
    trim_spectra, trim_wavelengths = spectra_subset(rest_wavelengths, spectra_to_fit,
                                     width=args.fit_width, line_cent=line_cent,
                                     lower_wave=args.lower_wave,
                                     upper_wave=args.upper_wave)

    ##Print out a summary of fitting settings
    print('Fitting using the following inputs:')
    print('\tfitting_model: {:s}'.format(args.fitting_model))
    print('\temission_line: {:s} ({:.1f} Angstrom)'.format(args.emission_line, line_cent))
    print('\twavelength bounds: {:.1f} to {:.1f} (Angstrom)'.format(trim_wavelengths[0], trim_wavelengths[-1]))

    ##Do the fit!
    fit = do_lmfit(lm_model, trim_wavelengths, trim_spectra, line_cent)
    ##Print results
    print(fit.fit_report())

    ##Plot fit if required
    if args.no_plots:
        pass
    else:
        do_fit_plot(wavelengths, spectra, trim_wavelengths, trim_spectra, fit)
