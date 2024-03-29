import numpy as np
import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, shift, rotate
from scipy.interpolate import NearestNDInterpolator

def plot_im(im, signal_i, signal_j, log=False, mask_edges=False):
    """
    Plotting function. plots a basic coronagraph image with crosshairs 
    at the specified location

    Parameters
    ----------
    im : 2D numpy array
        image you want to plot
    signal_i : int
        i location of the crosshairs to plot
    signal_j : int
        j location of the crosshairs to plot
    log : bool, optional
        True if you want a log plot. The default is False.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(20,20))
    
    if mask_edges:
        inner = 10 
        outer = 30
        imsz, imsz = im.shape
        imctr = (imsz-1)/2
        for i in range(imsz):
            for j in range(imsz):
                dist = np.sqrt((i-imctr)**2 + (j - imctr)**2)
                if dist < inner or dist > outer:
                    im[i, j] = np.nan
                
    
    if log:
        plt.imshow(np.log(im), origin='lower')
    else:
        plt.imshow(im, origin='lower')
    plt.axhline(signal_i, color="white", linestyle="--")
    plt.axvline(signal_j, color="white", linestyle="--")
    plt.colorbar()
    
    
def plot_im_ADI(im, im1_signal_i, im1_signal_j, im2_signal_i, im2_signal_j):
    """
    Plotting function. plots a basic coronagraph image with crosshairs 
    at the specified locations. This is useful for ADI subtracted images
    where the planet shows up in two locations separated by the roll angle

    Parameters
    ----------
    im : 2D numpy array
        image you want to plot
    im1_signal_i : int
        i location of the first crosshairs to plot
    im1_signal_j : int
        j location of the first crosshairs to plot
    im2_signal_i : int
        i location of the second crosshairs to plot
    im2_signal_j : int
        j location of the second crosshairs to plot

    Returns
    -------
    None.

    """
    plt.figure(figsize=(20,20))
    plt.imshow(im, origin='lower')
    plt.axhline(im1_signal_i, color="white", linestyle="--")
    plt.axvline(im1_signal_j, color="white", linestyle="--")
    plt.axhline(im2_signal_i, color="orange", linestyle="--")
    plt.axvline(im2_signal_j, color="orange", linestyle="--")
    plt.colorbar()
    

def high_pass_filter(img, filtersize=10):
    """
    A FFT implmentation of high pass filter.
    This function was adapted from the PyKlip suite
    (https://pyklip.readthedocs.io/en/latest/)

    Parameters
    ----------
        img: a 2D image
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Returns
    -------
        filtered: the filtered image
    """
    # mask NaNs if there are any
    nan_index = np.where(np.isnan(img))
    if np.size(nan_index) > 0:
        good_index = np.where(~np.isnan(img))
        y, x = np.indices(img.shape)
        good_coords = np.array([x[good_index], y[good_index]]).T # shape of Npix, ndimage
        nan_fixer = NearestNDInterpolator(good_coords, img[good_index])
        fixed_dat = nan_fixer(x[nan_index], y[nan_index])
        img[nan_index] = fixed_dat

    transform = np.fft.fft2(img)

    # coordinate system in FFT image
    u,v = np.meshgrid(np.fft.fftfreq(transform.shape[1]), np.fft.fftfreq(transform.shape[0]))
    # scale u,v so it has units of pixels in FFT space
    rho = np.sqrt((u*transform.shape[1])**2 + (v*transform.shape[0])**2)
    # scale rho up so that it has units of pixels in FFT space
    # rho *= transform.shape[0]
    # create the filter
    filt = 1. - np.exp(-(rho**2/filtersize**2))

    filtered = np.real(np.fft.ifft2(transform*filt))

    # restore NaNs
    filtered[nan_index] = np.nan
    img[nan_index] = np.nan

    return filtered



def calculate_cc_map(matched_filter_datacube, im, valid_mask, hipass=False, filtersize=10):
    """
    

    Parameters
    ----------
    matched_filter_datacube : 3D numpy array
        datacube containing all combinations of matched filters
    im : 2D numpy array
        image you want to apply the matched filter to
    valid_mask : 2D numpy array, boolean
        Where the matched filter is valid. i.e. only valid inside the OWA and
        outside the IWA
    hipass : bool, optional
        True if you want to apply a hipass filter first. The default is False.
    filtersize : float
        filter size for high pass filter. The default is 10.

    Returns
    -------
    cc_map : 2D numpy array
        map of matched filter (cross-correlation) values.

    """
    
    # cross-correlate
    Npix_i, Npix_j = im.shape
    cc_map = np.empty_like(im)
    if hipass:
        im = high_pass_filter(im, filtersize=filtersize)
    for i in range(Npix_i):
        for j in range(Npix_j):      
            if valid_mask[i, j]:
                matched_filter = matched_filter_datacube[i,j]    
                corr = np.nansum(matched_filter * im)
            else:
                corr = np.nan

            cc_map[i, j] = corr
    
    return cc_map

def calculate_mf_map(matched_filter_datacube, im, valid_mask, hipass=False, filtersize=10):
    """
    

    Parameters
    ----------
    matched_filter_datacube : 3D numpy array
        datacube containing all combinations of matched filters
    im : 2D numpy array
        image you want to apply the matched filter to
    valid_mask : 2D numpy array, boolean
        Where the matched filter is valid. i.e. only valid inside the OWA and
        outside the IWA
    hipass : bool, optional
        True if you want to apply a hipass filter first. The default is False.
    filtersize : float
        filter size for high pass filter. The default is 10.

    Returns
    -------
    cc_map : 2D numpy array
        map of matched filter (cross-correlation) values.

    """
    
    # cross-correlate
    Npix_i, Npix_j = im.shape
    mf_map = np.empty_like(im)
    if hipass:
        im = high_pass_filter(im, filtersize=filtersize)
    for i in range(Npix_i):
        for j in range(Npix_j):      
            if valid_mask[i, j]:
                matched_filter = matched_filter_datacube[i,j]    
                mf = 2 * np.nansum(matched_filter * im) / np.nansum(matched_filter**2)
            else:
                mf = np.nan

            mf_map[i, j] = mf
    
    return mf_map

def mas_to_lamD(sep_mas, lam, D):
    """
    function to convert mas to lam/D units

    Parameters
    ----------
    sep_mas : float
        value in mas. must have astropy.units.mas attached
    lam : float
        wavelength. must have astropy units
    D : float
        diameter of telescope. must have astropy.units.m attached

    Returns
    -------
    lamD_sep : float
        separation in lam/D units.

    """
    # sep_mas: planet--star separation in mas
    # lam: wl of observation in um
    # D: diam of telescope in m
    
    # returns lamD: separation in lam/D
    lam = lam.to(u.m)
    sep_lamD = (D/lam) * (1/u.radian) * sep_mas.to(u.radian)
    return sep_lamD

def lamD_to_mas(lamD_sep, lam, D):
    """
    function to convert lam/D to mas units

    Parameters
    ----------
    lamD_sep : float
        separation in lam/D units.
    lam : float
        wavelength. must have astropy units
    D : float
        diameter of telescope. must have astropy.units.m attached

    Returns
    -------
    sep_mas : float
        value in mas. must have astropy.units.mas attached

    """
    # sep_lamD: planet--star separation in lamD
    # lam: wl of observation in um
    # D: diam of telescope in m
    # returns sep_mas: separation in mas
    
    lam = lam.to(u.m)
    sep_mas = (lamD_sep * (lam / D) * u.radian).to(u.mas)
    
    return sep_mas

def construct_maps(arr, pixscale_mas, diam, IWA_lamD=8.5, OWA_lamD=26., plotting=False):
    """
    constructing maps showing (1) rotation values for each pixel, (2) a boolean
    array showing which pixels are inside/outside the OWA/IWA, and (3) a map of 
    how far away each pixel is from the image center

    Parameters
    ----------
    arr : 2D numpy array
        image
    pixscale_mas : float
        pixel scale of image
    diam : float
        diameter of telescope.
    IWA_lamD : float, optional
        IWA of telescope. The default is 8.5.
    OWA_lamD : float, optional
        OWA of telescope. The default is 26..
    plotting : bool, optional
        True if you want to plot maps. The default is False.

    Returns
    -------
    rotation_map : 2D numpy arrary
        rotation values for each pixel.
    valid_mask : 2D numpy arrary
        a boolean array showing which pixels are inside/outside the OWA/IWA
    radius_map : 2D numpy arrary
        a map of how far away each pixel is from the image center

    """
    Npix_i, Npix_j = arr.shape

    IWA_mas = lamD_to_mas(IWA_lamD, 0.5* u.um, diam*0.9*u.m)
    OWA_mas = lamD_to_mas(OWA_lamD, 0.5*u.um, diam*0.9*u.m)
    
    min_pix = IWA_mas.value / pixscale_mas
    max_pix = OWA_mas.value / pixscale_mas
    
    
    if (Npix_i % 2) == 0:
        center_i, center_j = Npix_j / 2. + 0.5, Npix_j / 2.+0.5
    else:
        center_i, center_j = (Npix_j - 1)/2,(Npix_j-1)/2

    if max_pix > center_i:
        max_pix = center_i -3

    rotation_map = np.empty((Npix_i, Npix_j))
    radius_map = np.empty((Npix_i, Npix_j))
    valid_mask = np.zeros_like(rotation_map, dtype=bool)

    for i in range(Npix_i):
        for j in range(Npix_j):
            angle = np.rad2deg(np.arctan2((i-center_i), (j-center_j)))
            rotation_map[i, j] = angle

            dist_from_center = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            radius_map[i, j] = dist_from_center
            if (dist_from_center >= min_pix) and (dist_from_center < max_pix):
                valid_mask[i, j] = True
    if plotting:
        plt.figure(figsize=(10, 10))
        plt.title("Rotation Map")
        plt.imshow(rotation_map, origin='lower')
        plt.colorbar()
        plt.figure(figsize=(10, 10))
        plt.title("Valid Pixels")
        plt.imshow(valid_mask, origin='lower')
        plt.colorbar()
        
    return rotation_map, valid_mask, radius_map

def get_psf_stamp(psf, psf_i, psf_j, pix_radius):
    """
    a function to get a small stamp image of the psf

    Parameters
    ----------
    psf : 2D numpy array
        large psf image.
    psf_i : int
        i location of psf
    psf_j : int
        j location of psf
    pix_radius : TYPE
        radius of desired psf stamp (in pixels)

    Returns
    -------
    psf_stamp : 2D numpy array
        smaller stamp image of psf.

    """
    x_min = int(psf_i-pix_radius)
    x_max = int(psf_i+pix_radius)+1
    
    y_min = int(psf_j-pix_radius)
    y_max = int(psf_j+pix_radius) +1
    psf_stamp = psf[x_min:x_max, y_min:y_max]
    stamp_center = pix_radius
    
    for i in range(pix_radius*2 + 1):
        for j in range(pix_radius*2 + 1):
            
            dist_from_center = np.sqrt((i-stamp_center)**2 + (j-stamp_center)**2)
            if dist_from_center > pix_radius:
                psf_stamp[i, j] = 0
    
    return psf_stamp





def synthesize_images_ADI3(im_dir, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, incl, zodis, aperture, roll_angle,
                          target_SNR=7, tot_tint=None, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, simple_planet=False, background=None, matched_filter_datacube_single=None):
    """
    function to synthesize images for ADI

    Parameters
    ----------
    im_dir : string
        directory that points to noiseless detector images.
    sci_plan_i : int
        i location of planet signal in first "science" image.
    sci_plan_j : int
        j location of planet signal in first "science" image.
    ref_plan_i : int
        i location of planet signal in second "reference" image.
    ref_plan_j : int
        j location of planet signal in second "reference" image.
    zodis : float
        zodi level of image.
    aperture : 2D numpy array, boolean
        aperture placed on image to measure signal/noise.
    target_SNR : float, optional
        injected SNR of signal. The default is 7.
    pix_radius : int, optional
        radius of aperture. The default is 1.
    verbose : bool, optional
        True if you want to print things. The default is False.
    planet_noise : bool
        True if you want to include planet noise. The default is True.
    add_noise : bool
        True if you want to include all noise except for planet noise. The default is True.
    add_star : bool
        True if you want to include the star. The default is True.
    uniform_disk : bool, optional
        true if you want a uniform exozodi disk. The default is False.
    simple_planet : bool, optional
        True if you don't want to include all of planet psf. The default is False.

    Returns
    -------
    science_image : 2D array
        noisy image 1 "science" image
    reference_image : 2D array
        noisy image 2 "reference" image of planet after ADI rotation
    sci_planet_counts : float
        photon count of the planet signal
    tot_noise_counts : float
        photon count of all noise sources at planet location
    tot_noise_counts_out : float
        photon count of all noise sources at location outside of exozodi structure
    (sci_out_i, sci_out_j) : tuple, int
        coordinates of a location outside of exozodi structure

    """
    sci_plan_im_fits = pyfits.open(im_dir + "/DET/sci_plan.fits")
    sci_plan_im = sci_plan_im_fits[0].data[0, 0]
    
    sci_star_im_fits = pyfits.open(im_dir + "/DET/sci_star.fits")
    sci_star_im = sci_star_im_fits[0].data[0,0]
    
    
    ref_plan_im_fits = pyfits.open(im_dir + "/DET/ref_plan.fits")
    ref_plan_im = ref_plan_im_fits[0].data[0, 0]
    
    ref_star_im_fits = pyfits.open(im_dir + "/DET/ref_star.fits")
    ref_star_im = ref_star_im_fits[0].data[0,0]
    
    
    
    
    sci_aperture_mask = np.zeros_like(sci_plan_im)
    sci_aperture_mask[sci_plan_i-pix_radius:sci_plan_i+pix_radius+1, sci_plan_j-pix_radius:sci_plan_j+pix_radius+1] = aperture
    
    ref_aperture_mask = np.zeros_like(ref_plan_im)
    ref_aperture_mask[ref_plan_i-pix_radius:ref_plan_i+pix_radius+1, ref_plan_j-pix_radius:ref_plan_j+pix_radius+1] = aperture
    
    if simple_planet is True:
        sci_plan_im = sci_plan_im*sci_aperture_mask
        ref_plan_im = ref_plan_im*ref_aperture_mask

    else:
        pass
    
    # define a location outside of the resonant ring
    if "LUVOIR-A" in im_dir:
        out_loc = 90
    elif "LUVOIR-B" in im_dir:
        out_loc = 80
    displacement = out_loc - sci_plan_j
    sci_out_i = sci_plan_i 
    sci_out_j = sci_plan_j + displacement
    ref_out_i = ref_plan_i + int(np.sqrt(displacement**2/2))
    ref_out_j = ref_plan_j + int(np.sqrt(displacement**2/2))
    
    
    sci_aperture_mask_out = np.zeros_like(sci_plan_im)
    sci_aperture_mask_out[sci_out_i-pix_radius:sci_out_i+pix_radius+1, sci_out_j-pix_radius:sci_out_j+pix_radius+1] = aperture
    
    ref_aperture_mask_out = np.zeros_like(ref_plan_im)
    ref_aperture_mask_out[ref_out_i-pix_radius:ref_out_i+pix_radius+1, ref_out_j-pix_radius:ref_out_j+pix_radius+1] = aperture

    sci_plan_CR = np.sum(sci_plan_im * sci_aperture_mask)
    ref_plan_CR = np.sum(ref_plan_im * ref_aperture_mask)
    
    
    
    sci_disk_im_fits = pyfits.open(im_dir + "/DET/sci_disk.fits")
    sci_disk_im = sci_disk_im_fits[0].data[0, 0]
    
    ref_disk_im_fits = pyfits.open(im_dir + "/DET/ref_disk.fits")
    ref_disk_im = ref_disk_im_fits[0].data[0, 0]
    
    sub_disk_im = sci_disk_im - ref_disk_im

    

    if uniform_disk:
        
        disk_val_at_planet = sci_disk_im[sci_plan_i, sci_plan_j]
        sci_disk_im = disk_val_at_planet * np.ones_like(sci_disk_im)
        ref_disk_im = disk_val_at_planet * np.ones_like(ref_disk_im)
        
    
    if add_star:
        pass
    else:
        sci_star_im = np.zeros_like(sci_star_im)
        ref_star_im = np.zeros_like(ref_star_im)
        
        
    
    sci_im_total = sci_plan_im + sci_star_im + sci_disk_im
    ref_im_total = ref_plan_im + ref_star_im + ref_disk_im
    
    sub_im_total = sci_im_total - ref_im_total
    
    
    
# =============================================================================
#     sci_sig_CR = sub_im_total * sci_aperture_mask
#     ref_sig_CR = sub_im_total * ref_aperture_mask
#     
#     signal_apertures = np.sum(sci_sig_CR) + -1*np.sum(ref_sig_CR)
# =============================================================================
    
    sci_sig_CR = sci_plan_im * sci_aperture_mask
    ref_sig_CR = ref_plan_im * ref_aperture_mask
    
    
    signal_apertures = np.sum(sci_sig_CR) + np.sum(ref_sig_CR)
    
    
    
    sci_signal_i_opp, sci_signal_j_opp  = get_opp_coords(sci_plan_i, sci_plan_j, (101-1)/2)
    ref_signal_i_opp, ref_signal_j_opp  = get_opp_coords(ref_plan_i, ref_plan_j, (101-1)/2)
    
    if background == "region":
        inner_r = 2
        outer_r = 6
        
        ## define noise region

        nr_dynasquare_sci = region(sci_im_total, sci_plan_i, sci_plan_j, aperture, pix_radius, inner_r, outer_r, opposite=False)
        #nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, ref_im_total, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, inner_r, aperture, pix_radius, roll_angle, opposite=False)
        nr_dynasquare_ref = region(ref_im_total, ref_plan_i, ref_plan_j, aperture, pix_radius, inner_r, outer_r, opposite=False)

        
        ## measure noise
        counts_per_ap_nr_dynasquare_sci_photometry, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, pix_radius)
        counts_per_ap_nr_dynasquare_ref_photometry, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, pix_radius)
        
        
        # get aperture photometry noise
        noise_background_ap = np.mean(counts_per_ap_nr_dynasquare_sci_photometry) + np.mean(counts_per_ap_nr_dynasquare_ref_photometry)


        # get matched filter noise
        mf_noises_sci = []
        mf_noises_ref = []

        for ap_coord in ap_coords_nr_dynasquare_sci:
            i, j = ap_coord
            mf_single_sci = np.nansum(matched_filter_datacube_single[i, j] * nr_dynasquare_sci) / np.nansum(matched_filter_datacube_single[i, j]**2)
            mf_noises_sci.append(mf_single_sci)
        for ap_coord in ap_coords_nr_dynasquare_ref:
            i, j = ap_coord
            mf_single_ref = np.nansum(matched_filter_datacube_single[i, j] * nr_dynasquare_ref) / np.nansum(matched_filter_datacube_single[i, j]**2)
            mf_noises_ref.append(mf_single_ref)
        mf_noises_sci = np.array(mf_noises_sci)
        mf_noises_ref = np.array(mf_noises_ref)
        #noise_background = np.median(mf_noises_sci) + np.median(mf_noises_ref)
        #noise_background_mf = np.nanmean(mf_noises_sci) + np.nanmean(mf_noises_ref)
        noise_background_mf = noise_background_ap
        
 

        
        
    elif background == "planetloc":
        
        sci_star_CR = np.sum(sci_star_im * sci_aperture_mask)
        ref_star_CR = np.sum(ref_star_im * ref_aperture_mask)
        
        sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
        ref_disk_CR = np.sum(ref_disk_im*ref_aperture_mask)
        
        sci_bkgr_CR = sci_disk_CR + sci_star_CR
        ref_bkgr_CR = ref_disk_CR + ref_star_CR
        
        noise_background_mf = sci_bkgr_CR + ref_bkgr_CR
        noise_background_ap = sci_bkgr_CR + ref_bkgr_CR

        
    else:
        assert False, "Please specify method for background calculation"
        
    
    total_signal_CR = signal_apertures  
# =============================================================================
#     if uniform_disk is False and incl == 0 and (zodis < 10):
#         total_signal_CR  = total_signal_CR + np.sum(sub_disk_im*sci_aperture_mask) + -1*np.sum(sub_disk_im*ref_aperture_mask)
# =============================================================================
    total_noise_mf_CR = noise_background_mf + total_signal_CR
    total_bkgr_mf_CR = noise_background_mf
    
    total_noise_ap_CR = noise_background_ap + total_signal_CR
    total_bkgr_ap_CR = noise_background_ap
    
    if target_SNR is None and tot_tint is None:
        assert False, "target_SNR and tot_tint are both None. Set one to something."
    elif target_SNR is not None and tot_tint is not None:
        assert False, "target_SNR and tot_tint cannot both be set!"
    elif target_SNR is not None and tot_tint is None:
        tot_tint_mf = target_SNR**2 * total_noise_mf_CR / total_signal_CR**2
        tot_tint_ap = target_SNR**2 * total_noise_ap_CR / total_signal_CR**2

    elif target_SNR is None and tot_tint is not None:
        tot_tint_mf = tot_tint
        tot_tint_ap = tot_tint
        # ap and mf images will end up being the same
        pass
    else:
        assert False, "Something is wrong... check target_SNR and tot_tint"
        
    

    
    if verbose:
        print("signal CR:", total_signal_CR)
        print("noise CR MF:", total_noise_mf_CR)
        print("Total Signal Counts:", total_signal_CR*tot_tint_mf)
        print("Total Background Counts MF:", (total_bkgr_mf_CR )*tot_tint_mf)
        
        print("Total Noise Counts MF:", (total_noise_mf_CR )*tot_tint_mf)
 
        
        print("tot_tint_mf", tot_tint_mf)
        SNR_calc_mf = total_signal_CR * tot_tint_mf / np.sqrt(total_noise_mf_CR*tot_tint_mf)
        print("SNR_calc_mf:", SNR_calc_mf)
        
        print("###########")
        
        print("signal CR:", total_signal_CR)
        print("noise CR AP:", total_noise_ap_CR)
        
        print("Total Signal Counts:", total_signal_CR*tot_tint_ap)
        print("Total Background Counts AP:", (total_bkgr_ap_CR )*tot_tint_ap)
        
        print("Total Noise Counts AP:", (total_noise_ap_CR )*tot_tint_ap)
 
        
        print("tot_tint_ap", tot_tint_ap)
        SNR_calc_ap = total_signal_CR * tot_tint_ap / np.sqrt(total_noise_ap_CR*tot_tint_ap)
        print("SNR_calc_ap:", SNR_calc_ap)
        
        
        
    
    
    tot_noise_counts_ap = total_noise_ap_CR*tot_tint_ap
    tot_bkgr_counts_ap = total_bkgr_ap_CR * tot_tint_ap
    
    tot_noise_counts_mf = total_noise_mf_CR*tot_tint_mf
    tot_bkgr_counts_mf = total_bkgr_mf_CR * tot_tint_mf
    
    




    science_image_ap = (sci_plan_im + sci_disk_im + sci_star_im) * tot_tint_ap
    reference_image_ap = (ref_plan_im + ref_disk_im + ref_star_im) * tot_tint_ap
    
    science_image_mf = (sci_plan_im + sci_disk_im + sci_star_im) * tot_tint_mf
    reference_image_mf = (ref_plan_im + ref_disk_im + ref_star_im) * tot_tint_mf

    sub_disk_im_noiseless = (sci_disk_im + sci_star_im) * tot_tint_mf - (ref_disk_im + ref_star_im) * tot_tint_mf
    
    science_image_noisy_ap = np.random.poisson(science_image_ap)
    reference_image_noisy_ap = np.random.poisson(reference_image_ap)
    
    science_image_noisy_mf = np.random.poisson(science_image_mf)
    reference_image_noisy_mf = np.random.poisson(reference_image_mf)
    
    
    if add_noise:
        science_image_ap = science_image_noisy_ap.astype(float)
        reference_image_ap = reference_image_noisy_ap.astype(float)
        
        science_image_mf = science_image_noisy_mf.astype(float)
        reference_image_mf = reference_image_noisy_mf.astype(float)
    
    

    return science_image_ap, reference_image_ap, science_image_mf, reference_image_mf, \
            tot_noise_counts_ap, tot_noise_counts_mf, \
            tot_bkgr_counts_ap, tot_bkgr_counts_mf, \
            (sci_out_i, sci_out_j, ref_out_i, ref_out_j), \
            tot_tint_ap, tot_tint_mf, \
            sub_disk_im_noiseless


    
  
    

def get_CR(im_dir, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, incl, zodis, aperture, roll_angle,
                          target_SNR=7, tot_tint=None, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, simple_planet=False, background=None):
    """
    function to synthesize images for ADI

    Parameters
    ----------
    im_dir : string
        directory that points to noiseless detector images.
    sci_plan_i : int
        i location of planet signal in first "science" image.
    sci_plan_j : int
        j location of planet signal in first "science" image.
    ref_plan_i : int
        i location of planet signal in second "reference" image.
    ref_plan_j : int
        j location of planet signal in second "reference" image.
    zodis : float
        zodi level of image.
    aperture : 2D numpy array, boolean
        aperture placed on image to measure signal/noise.
    target_SNR : float, optional
        injected SNR of signal. The default is 7.
    pix_radius : int, optional
        radius of aperture. The default is 1.
    verbose : bool, optional
        True if you want to print things. The default is False.
    planet_noise : bool
        True if you want to include planet noise. The default is True.
    add_noise : bool
        True if you want to include all noise except for planet noise. The default is True.
    add_star : bool
        True if you want to include the star. The default is True.
    uniform_disk : bool, optional
        true if you want a uniform exozodi disk. The default is False.
    simple_planet : bool, optional
        True if you don't want to include all of planet psf. The default is False.

    Returns
    -------
    science_image : 2D array
        noisy image 1 "science" image
    reference_image : 2D array
        noisy image 2 "reference" image of planet after ADI rotation
    sci_planet_counts : float
        photon count of the planet signal
    tot_noise_counts : float
        photon count of all noise sources at planet location
    tot_noise_counts_out : float
        photon count of all noise sources at location outside of exozodi structure
    (sci_out_i, sci_out_j) : tuple, int
        coordinates of a location outside of exozodi structure

    """
    sci_plan_im_fits = pyfits.open(im_dir + "/DET/sci_plan.fits")
    sci_plan_im = sci_plan_im_fits[0].data[0, 0]
    
    sci_star_im_fits = pyfits.open(im_dir + "/DET/sci_star.fits")
    sci_star_im = sci_star_im_fits[0].data[0,0]
    
    
    ref_plan_im_fits = pyfits.open(im_dir + "/DET/ref_plan.fits")
    ref_plan_im = ref_plan_im_fits[0].data[0, 0]
    
    ref_star_im_fits = pyfits.open(im_dir + "/DET/ref_star.fits")
    ref_star_im = ref_star_im_fits[0].data[0,0]
    
    
    
    
    sci_aperture_mask = np.zeros_like(sci_plan_im)
    sci_aperture_mask[sci_plan_i-pix_radius:sci_plan_i+pix_radius+1, sci_plan_j-pix_radius:sci_plan_j+pix_radius+1] = aperture
    
    ref_aperture_mask = np.zeros_like(ref_plan_im)
    ref_aperture_mask[ref_plan_i-pix_radius:ref_plan_i+pix_radius+1, ref_plan_j-pix_radius:ref_plan_j+pix_radius+1] = aperture
    
    if simple_planet is True:
        sci_plan_im = sci_plan_im*sci_aperture_mask
        ref_plan_im = ref_plan_im*ref_aperture_mask

    else:
        pass
    
    # define a location outside of the resonant ring
    if "LUVOIR-A" in im_dir:
        out_loc = 90
    elif "LUVOIR-B" in im_dir:
        out_loc = 80
    displacement = out_loc - sci_plan_j
    sci_out_i = sci_plan_i 
    sci_out_j = sci_plan_j + displacement
    ref_out_i = ref_plan_i + int(np.sqrt(displacement**2/2))
    ref_out_j = ref_plan_j + int(np.sqrt(displacement**2/2))
    
    
    sci_aperture_mask_out = np.zeros_like(sci_plan_im)
    sci_aperture_mask_out[sci_out_i-pix_radius:sci_out_i+pix_radius+1, sci_out_j-pix_radius:sci_out_j+pix_radius+1] = aperture
    
    ref_aperture_mask_out = np.zeros_like(ref_plan_im)
    ref_aperture_mask_out[ref_out_i-pix_radius:ref_out_i+pix_radius+1, ref_out_j-pix_radius:ref_out_j+pix_radius+1] = aperture

    sci_plan_CR = np.sum(sci_plan_im * sci_aperture_mask)
    ref_plan_CR = np.sum(ref_plan_im * ref_aperture_mask)
    
    
    
    sci_disk_im_fits = pyfits.open(im_dir + "/DET/sci_disk.fits")
    sci_disk_im = sci_disk_im_fits[0].data[0, 0]
    
    ref_disk_im_fits = pyfits.open(im_dir + "/DET/ref_disk.fits")
    ref_disk_im = ref_disk_im_fits[0].data[0, 0]
    
    sub_disk_im = sci_disk_im - ref_disk_im

    

    if uniform_disk:
        
        disk_val_at_planet = sci_disk_im[sci_plan_i, sci_plan_j]
        sci_disk_im = disk_val_at_planet * np.ones_like(sci_disk_im)
        ref_disk_im = disk_val_at_planet * np.ones_like(ref_disk_im)
        
    
    if add_star:
        pass
    else:
        sci_star_im = np.zeros_like(sci_star_im)
        ref_star_im = np.zeros_like(ref_star_im)
        
        
    
    sci_im_total = sci_plan_im + sci_star_im + sci_disk_im
    ref_im_total = ref_plan_im + ref_star_im + ref_disk_im
    
    sub_im_total = sci_im_total - ref_im_total
    
    
    
    sci_sig_CR = sci_plan_im * sci_aperture_mask
    ref_sig_CR = ref_plan_im * ref_aperture_mask
    
    signal_apertures = np.sum(sci_sig_CR) + np.sum(ref_sig_CR)
    
    
    if background == "region":
        inner_r = 2
        outer_r = 6
        
        ## define noise region
        nr_dynasquare_sci = region(sci_im_total, sci_plan_i, sci_plan_j, aperture, pix_radius, inner_r, outer_r, opposite=False)
        nr_dynasquare_ref = region(ref_im_total, ref_plan_i, ref_plan_j, aperture, pix_radius, inner_r, outer_r, opposite=False)
    
        
        ## measure noise
        counts_per_ap_nr_dynasquare_sci_photometry, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, pix_radius)
        counts_per_ap_nr_dynasquare_ref_photometry, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, pix_radius)
        
        
        # get aperture photometry noise
        noise_background = np.mean(counts_per_ap_nr_dynasquare_sci_photometry) + np.mean(counts_per_ap_nr_dynasquare_ref_photometry)
        
        
        ## define noise region
        nr_dynasquare_sci = region(sci_disk_im, sci_plan_i, sci_plan_j, aperture, pix_radius, inner_r, outer_r, opposite=False)
        nr_dynasquare_ref = region(ref_disk_im, ref_plan_i, ref_plan_j, aperture, pix_radius, inner_r, outer_r, opposite=False)
    
        
        ## measure noise
        counts_per_ap_nr_dynasquare_sci_photometry, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, pix_radius)
        counts_per_ap_nr_dynasquare_ref_photometry, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, pix_radius)
        
        noise_disk = np.mean(counts_per_ap_nr_dynasquare_sci_photometry) + np.mean(counts_per_ap_nr_dynasquare_ref_photometry)
        
        
        
    elif background == "planetloc":
        
        sci_star_CR = np.sum(sci_star_im * sci_aperture_mask)
        ref_star_CR = np.sum(ref_star_im * ref_aperture_mask)
        
        sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
        ref_disk_CR = np.sum(ref_disk_im*ref_aperture_mask)
        
        sci_bkgr_CR = sci_disk_CR + sci_star_CR
        ref_bkgr_CR = ref_disk_CR + ref_star_CR
        
        noise_background = sci_bkgr_CR + ref_bkgr_CR
        noise_disk = sci_disk_CR + ref_disk_CR

    
    total_signal_CR = signal_apertures  
    if uniform_disk is False and incl == 0 and (zodis < 10):
        total_signal_CR  = total_signal_CR + np.sum(sub_disk_im*sci_aperture_mask) + -1*np.sum(sub_disk_im*ref_aperture_mask)
    total_noise_CR = noise_background + total_signal_CR
    
    return total_signal_CR, total_noise_CR, noise_disk
    


def synthesize_images_RDI3(im_dir, sci_plan_i, sci_plan_j, zodis, aperture,
                          target_SNR=7, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, r2_disk=False, simple_planet=False, zerodisk=False, background=None, matched_filter_datacube_single=None):
    """
    function to synthesize images for RDI

    Parameters
    ----------
    im_dir : string
        directory that points to noiseless detector images.
    sci_plan_i : int
        i location of planet signal in first "science" image.
    sci_plan_j : int
        j location of planet signal in first "science" image.
    zodis : float
        zodi level of image.
    aperture : 2D numpy array, boolean
        aperture placed on image to measure signal/noise.
    target_SNR : float, optional
        injected SNR of signal. The default is 7.
    pix_radius : int, optional
        radius of aperture. The default is 1.
    verbose : bool, optional
        True if you want to print things. The default is False.
    planet_noise : bool
        True if you want to include planet noise. The default is True.
    add_noise : bool
        True if you want to include all noise except for planet noise. The default is True.
    add_star : bool
        True if you want to include the star. The default is True.
    uniform_disk : bool, optional
        true if you want a uniform exozodi disk. The default is False.
    simple_planet : bool, optional
        True if you don't want to include all of planet psf. The default is False.

    Returns
    -------
    science_image : 2D array
        noisy image 1 "science" image
    reference_image : 2D array
        noisy image 2 "reference" image of planet after ADI rotation
    sci_planet_counts : float
        photon count of the planet signal
    tot_noise_counts : float
        photon count of all noise sources at planet location
    tot_noise_counts_out : float
        photon count of all noise sources at location outside of exozodi structure
    (sci_out_i, sci_out_j) : tuple, int
        coordinates of a location outside of exozodi structure

    """
    
    sci_plan_im_fits = pyfits.open(im_dir + "/DET/sci_plan.fits")
    sci_plan_im = sci_plan_im_fits[0].data[0, 0]
    
    sci_star_im_fits = pyfits.open(im_dir + "/DET/sci_star.fits")
    sci_star_im = sci_star_im_fits[0].data[0,0]
    
    ref_star_im_fits = pyfits.open(im_dir + "/DET/ref_star.fits")
    ref_star_im = ref_star_im_fits[0].data[0,0]
    

    
    
    sci_aperture_mask = np.zeros_like(sci_plan_im)
    sci_aperture_mask[sci_plan_i-pix_radius:sci_plan_i+pix_radius+1, sci_plan_j-pix_radius:sci_plan_j+pix_radius+1] = aperture
    
    
    if simple_planet is True:
        sci_plan_im = sci_plan_im*sci_aperture_mask

    else:
        pass
    
    # define a location outside of the resonant ring
    if "LUVOIR-A" in im_dir:
        out_loc = 90
    elif "LUVOIR-B" in im_dir:
        out_loc = 80
    displacement = out_loc - sci_plan_j
    sci_out_i = sci_plan_i 
    sci_out_j = sci_plan_j + displacement
    
    
    sci_aperture_mask_out = np.zeros_like(sci_plan_im)
    sci_aperture_mask_out[sci_out_i-pix_radius:sci_out_i+pix_radius+1, sci_out_j-pix_radius:sci_out_j+pix_radius+1] = aperture
    
        
    
        
    sci_disk_im_fits = pyfits.open(im_dir + "/DET/sci_disk.fits")
    sci_disk_im = sci_disk_im_fits[0].data[0, 0]
    
    if uniform_disk:
        
        disk_val_at_planet = sci_disk_im[sci_plan_i, sci_plan_j]
        sci_disk_im = disk_val_at_planet * np.ones_like(sci_disk_im)
        
    
    if zerodisk:
        sci_disk_im = np.zeros_like(sci_disk_im)
    
    if add_star:
        pass
    else:
        sci_star_im = np.zeros_like(sci_star_im)
        ref_star_im = np.zeros_like(ref_star_im)
    

    
    sci_im_total = sci_plan_im + sci_star_im + sci_disk_im
    ref_im_total = ref_star_im 
    
    sub_im_total = sci_im_total - ref_im_total
    
    
    
    sci_sig_CR = sci_plan_im * sci_aperture_mask
    
    signal_apertures = np.sum(sci_sig_CR)
    

    
    
    
    sci_signal_i_opp, sci_signal_j_opp  = get_opp_coords(sci_plan_i, sci_plan_j, (101-1)/2)
    
    if background == "region":
        inner_r = 2
        outer_r = 6
        

    
        nr_dynasquare_sci = region(sci_im_total, sci_plan_i, sci_plan_j, aperture, pix_radius, inner_r, outer_r, opposite=False)
        nr_dynasquare_ref = region(ref_im_total, sci_plan_i, sci_plan_j, aperture, pix_radius, inner_r, outer_r, opposite=False)

        ## measure noise
        counts_per_ap_nr_dynasquare_sci_photometry, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, pix_radius)
        counts_per_ap_nr_dynasquare_ref_photometry, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, pix_radius)
        
        tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci_photometry# - np.median(counts_per_ap_nr_dynasquare_sci)
        tot_ref_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_ref_photometry# - np.median(counts_per_ap_nr_dynasquare_ref)    

        noise_background_ap = np.median(tot_sci_ap_counts_dynasquare) + np.median(tot_ref_ap_counts_dynasquare)


        # get matched filter noise
        mf_noises_sci = []
        mf_noises_ref = []

        for ap_coord in ap_coords_nr_dynasquare_sci:
            i, j = ap_coord
            mf_single_sci = np.nansum(matched_filter_datacube_single[i, j] * nr_dynasquare_sci) / np.nansum(matched_filter_datacube_single[i, j]**2)
            mf_noises_sci.append(mf_single_sci)
        for ap_coord in ap_coords_nr_dynasquare_ref:
            i, j = ap_coord
            mf_single_ref = np.nansum(matched_filter_datacube_single[i, j] * nr_dynasquare_ref) / np.nansum(matched_filter_datacube_single[i, j]**2)
            mf_noises_ref.append(mf_single_ref)
        mf_noises_sci = np.array(mf_noises_sci)
        mf_noises_ref = np.array(mf_noises_ref)
        
        #noise_background_mf = np.nanmean(mf_noises_sci) + np.nanmean(mf_noises_ref)
        noise_background_mf = noise_background_ap
    
    elif background == "planetloc":

        sci_star_CR = np.sum(sci_star_im * sci_aperture_mask)
        ref_star_CR = np.sum(ref_star_im * sci_aperture_mask)
        
        sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
        
        sci_bkgr_CR = sci_disk_CR + sci_star_CR
        ref_bkgr_CR = ref_star_CR
        
        noise_background_mf = sci_bkgr_CR + ref_bkgr_CR
        noise_background_ap = sci_bkgr_CR + ref_bkgr_CR

    else:
        assert False, "Please specify background method to use"
    


    total_signal_CR = signal_apertures
    total_noise_mf_CR = noise_background_mf + total_signal_CR
    total_bkgr_mf_CR = noise_background_mf
    
    total_noise_ap_CR = noise_background_ap + total_signal_CR
    total_bkgr_ap_CR = noise_background_ap
    
    tot_tint_ap = target_SNR**2 * total_noise_ap_CR / total_signal_CR**2
    tot_tint_mf = target_SNR**2 * total_noise_mf_CR / total_signal_CR**2

    
    
    
    if verbose:
        print("Total Signal Counts:", total_signal_CR*tot_tint_mf)
        print("Total Background Counts:", (total_bkgr_mf_CR )*tot_tint_mf)
        print("Total Noise Counts:", (total_noise_mf_CR )*tot_tint_mf)
 
        print("tot_tint_mf", tot_tint_mf)
        SNR_calc_mf = total_signal_CR * tot_tint_mf / np.sqrt(total_noise_mf_CR*tot_tint_mf)
        print("SNR_calc_mf:", SNR_calc_mf)
        
        print("###########")
        
        print("Total Signal Counts:", total_signal_CR*tot_tint_ap)
        print("Total Background Counts:", (total_bkgr_ap_CR )*tot_tint_ap)
        print("Total Noise Counts:", (total_noise_ap_CR )*tot_tint_ap)
 
        print("tot_tint_ap", tot_tint_ap)
        SNR_calc_ap = total_signal_CR * tot_tint_ap / np.sqrt(total_noise_ap_CR*tot_tint_ap)
        print("SNR_calc_ap:", SNR_calc_ap)
        
        
    
    
    tot_noise_counts_ap = total_noise_ap_CR*tot_tint_ap
    tot_bkgr_counts_ap = total_bkgr_ap_CR * tot_tint_ap
    
    tot_noise_counts_mf = total_noise_mf_CR*tot_tint_mf
    tot_bkgr_counts_mf = total_bkgr_mf_CR * tot_tint_mf



    science_image_ap = (sci_plan_im + sci_disk_im + sci_star_im) * tot_tint_ap
    reference_image_ap = ref_star_im * tot_tint_ap
    
    science_image_mf = (sci_plan_im + sci_disk_im + sci_star_im) * tot_tint_mf
    reference_image_mf = ref_star_im * tot_tint_mf
    
    sub_disk_im_noiseless = (sci_disk_im + sci_star_im) * tot_tint_mf - (ref_star_im) * tot_tint_mf

    
    science_image_noisy_ap = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * tot_tint_ap)
    reference_image_noisy_ap = np.random.poisson(ref_star_im * tot_tint_ap)
    
    science_image_noisy_mf = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * tot_tint_mf)
    reference_image_noisy_mf = np.random.poisson(ref_star_im * tot_tint_mf)
    
    
    
    if add_noise:
        science_image_ap = science_image_noisy_ap.astype(float)
        reference_image_ap = reference_image_noisy_ap.astype(float)
        
        science_image_mf = science_image_noisy_mf.astype(float)
        reference_image_mf = reference_image_noisy_mf.astype(float)
    
    

    return science_image_ap, reference_image_ap, science_image_mf, reference_image_mf, \
            tot_noise_counts_ap, tot_noise_counts_mf, tot_bkgr_counts_ap, tot_bkgr_counts_mf, \
                (sci_out_i, sci_out_j),  \
                tot_tint_ap, tot_tint_mf, \
                sub_disk_im_noiseless

    
    
    


def downbin_psf(psf, imsc, imsz, wave, diam, tele):
    """
    Downbins a stellar/planetary psf to scale defined by lam/D

    Parameters
    ----------
    psf : 2D array
        image of the psf you want to downbin.
    imsc : float
        pixel scale of psf image.
    imsz : int
        size of the image.
    wave : float
        wavelength of psf.
    diam : float
        diameter of telescope.
    tele : string
        name of telescope either "LUVA" or "LUVB".

    Returns
    -------
    temp : 2D array
        downbinned psf.

    """
    
    if tele == "LUVA":
        shift_order = 1
        zoom_order = 5
    elif tele == "LUVB":
        shift_order =1
        zoom_order = 5
    
    rad2mas = 180./np.pi*3600.*1000.

    imsc2 = 0.5*wave*1e-6/(0.9*diam)*rad2mas # mas
    
    # Compute wavelength-dependent zoom factor.
    fact = 0.25*wave * 1e-6 /(0.9*diam)*rad2mas/imsc2



    norm = np.sum(psf)

    # Scale image to imsc.
    temp = np.exp(zoom(np.log(psf), fact, mode='nearest', order=zoom_order)) # interpolate in log-space to avoid negative values
    
    
    temp *= norm/np.sum(temp) # ph/s
    
    
    
    # Center image so that (imsz-1)/2 is center.
    if (((temp.shape[0] % 2 == 0) and (imsz % 2 != 0)) or ((temp.shape[0] % 2 != 0) and (imsz % 2 == 0))):

        temp = np.pad(temp, ((0, 1), (0, 1)), mode='edge')

        temp = np.exp(shift(np.log(temp), (0.5, 0.5), order=shift_order)) # interpolate in log-space to avoid negative values


        temp = temp[1:-1, 1:-1]



    
    # Crop image to imsz.
    if (temp.shape[0] > imsz):
        nn = (temp.shape[0]-imsz)//2
        temp = temp[nn:-nn, nn:-nn]
    else:
        nn = (imsz-temp.shape[0])//2
        temp = np.pad(temp, ((nn, nn), (nn, nn)), mode='edge')
        
    return temp


def region_dynasquare(sub_im, signal_i, signal_j, aperture, ap_sz, width, height, opposite=False):
    """
    defines a region to be used to calculate noise

    Parameters
    ----------
    sub_im : 2D array
        psf subtracted image.
    signal_i : int
        i coord of signal.
    signal_j : int
        j coord of signal.
    aperture : 2D array
        mask of aperture used to measure signal/noise.
    ap_sz : int
        radius in pixels of aperture.
    width : int
        width of region used to calculate noise.
    height : int
        height of region used to calculate noise.
    opposite : bool, optional
        True if you want the region on the opposite side of the image as the signal. The default is False.

    Returns
    -------
    noise_region : 2D array
        region used to calculate noise.

    """
    
    imsz, imsz = sub_im.shape            

    noise_mask = np.zeros_like(sub_im)
    
    noise_mask[signal_i-width:signal_i+width+1, signal_j-height:signal_j+height+1] = 1
    
    
    if opposite:
        pass
    else:
        noise_mask[signal_i-ap_sz:signal_i+ap_sz+1, signal_j-ap_sz:signal_j+ap_sz+1] = ~aperture


    zero_inds = np.where(noise_mask == 0.)
    noise_mask[zero_inds] = np.nan
    
    noise_region = sub_im * noise_mask
    
    return noise_region

def region(sub_im, signal_i, signal_j, aperture, ap_sz, inner_r, outer_r, opposite=False):
    """
    defines a region to be used to calculate noise

    Parameters
    ----------
    sub_im : 2D array
        psf subtracted image.
    signal_i : int
        i coord of signal.
    signal_j : int
        j coord of signal.
    aperture : 2D array
        mask of aperture used to measure signal/noise.
    ap_sz : int
        radius in pixels of aperture.
    width : int
        width of region used to calculate noise.
    height : int
        height of region used to calculate noise.
    opposite : bool, optional
        True if you want the region on the opposite side of the image as the signal. The default is False.

    Returns
    -------
    noise_region : 2D array
        region used to calculate noise.

    """
    
    imsz, imsz = sub_im.shape            

    noise_mask = np.zeros_like(sub_im)
    
    for i in range(imsz):
        for j in range(imsz):
            dist = np.sqrt((signal_i-i)**2 + (signal_j-j)**2)
            if dist > inner_r and dist <= outer_r:
                noise_mask[i, j] = 1
                
      
    zero_inds = np.where(noise_mask == 0.)
    noise_mask[zero_inds] = np.nan
    
    noise_region = sub_im * noise_mask
    
    return noise_region



def get_opposite_dynasquare_region(dynasquare_region, im, signal_i_opp, signal_j_opp, ap_sz):
    
    imsz, imsz = dynasquare_region.shape
    imctr = (imsz-1)/2

    region_inds = ~np.isnan(dynasquare_region)
    opp_region = np.zeros_like(dynasquare_region)
    for i in range(imsz):
        for j in range(imsz):
            if region_inds[i,j]:
                i_opp = int(imctr - (i-imctr))
                j_opp = int(imctr - (j-imctr))
                opp_region[i_opp, j_opp] = 1
    opp_region[signal_i_opp-ap_sz:signal_i_opp+ap_sz+1, signal_j_opp-ap_sz:signal_j_opp+ap_sz+1] = 1
    
    zero_inds = np.where(opp_region == 0.)
    opp_region[zero_inds] = np.nan
    
    opp_noise_region = im * opp_region
    
    return opp_noise_region
    

def sum_apertures_in_region(noise_region, aperture, ap_sz):
    
    nan_map = ~np.isnan(noise_region)
    
    noise_region_median = np.nanmedian(noise_region)
    
    noise_region_bkgr_rm = noise_region #- noise_region_median
    
    
    imsz, imsz = noise_region.shape
    Npix_ap = np.sum(aperture)

    apertures_sampled = 0
    aperture_coords = []

    counts_per_aperture = []

    for i in range(imsz):
        for j in range(imsz):
            # aperture in question
            if nan_map[i,j]:
                noise_aperture = nan_map[i-ap_sz:i+ap_sz+1,j-ap_sz:j+ap_sz+1]
                if np.sum(noise_aperture[aperture]) == Npix_ap:
                    #image[i-ap_sz:i+ap_sz+1,j-ap_sz:j+ap_sz+1] = aperture
                    nan_map[i-ap_sz:i+ap_sz+1,j-ap_sz:j+ap_sz+1] = ~aperture & nan_map[i-ap_sz:i+ap_sz+1,j-ap_sz:j+ap_sz+1]
                    
                    noise_aperture = noise_region_bkgr_rm[i-ap_sz:i+ap_sz+1,j-ap_sz:j+ap_sz+1]
                    #print(noise_aperture[aperture])
                    counts_per_aperture.append(np.sum(noise_aperture[aperture]))
                    apertures_sampled += 1
                    aperture_coords.append([i, j])

    aperture_coords = np.array(aperture_coords)
    counts_per_aperture = np.array(counts_per_aperture)
    
    return counts_per_aperture, aperture_coords
    
def get_opp_coords(i, j, imctr):
    i_opp = imctr - (i-imctr)
    j_opp = imctr - (j-imctr)
    return int(i_opp), int(j_opp)  

            

def sigma_clip(arr, thresh=3):
    # sigma clip
    inds_higher = np.where(arr > np.median(arr) + thresh*np.std(arr, ddof=1))
    inds_lower = np.where(arr < np.median(arr) - thresh*np.std(arr, ddof=1))
    arr[inds_higher] = np.nan
    arr[inds_lower] = np.nan
    
    return arr
 


def r2_correction(noise_region):
    
    imsz, imsz = noise_region.shape
    imctr = (imsz-1)/2
    
    # get distribution of values by distance from center
    dists = []
    vals = []
    
    for i in range(imsz):
        for j in range(imsz):
            if ~np.isnan(noise_region[i,j]):
                # distance from center
                dist = np.sqrt((i-imctr)**2 + (j-imctr)**2)
                val = noise_region[i,j]
                dists.append(dist)
                vals.append(val)
                
    #sig_dist = np.sqrt((sig_i-imctr)**2 + (sig_j-imctr)**2) 
    
    
    dists = np.array(dists)

    coeffs = np.polyfit(dists, vals, 2)
# =============================================================================
#     x_arr = np.arange(np.min(dists), np.max(dists), 0.01)
#     y_fit = coeffs[2] + coeffs[1]*x_arr + coeffs[0]*x_arr**2 
#     plt.figure()
#     plt.scatter(dists, vals)
#     #plt.axvline(sig_dist, color="k")
#     plt.plot(x_arr, y_fit, color="C1")
#     assert False
# =============================================================================

    r2_corrected_region = np.copy(noise_region)
    for i in range(imsz):
        for j in range(imsz):
            if ~np.isnan(noise_region[i,j]):
                # distance from center
                dist = np.sqrt((i-imctr)**2 + (j-imctr)**2)
                r2_fit = coeffs[2] + coeffs[1]*dist + coeffs[0]*dist**2 
                r2_corrected_region[i,j] -= r2_fit
                
                # should I normalize to the value at the planet location??
    
    return r2_corrected_region

def az_correction(noise_region, rotation_map):
    rotation_map_copy = np.copy(rotation_map)
    rotation_map_copy += 270
    
    imsz, imsz = noise_region.shape
    imctr = (imsz-1)/2
    
    # get distribution of values by angle
    angs = []
    vals = []
    
    for i in range(imsz):
        for j in range(imsz):
            if ~np.isnan(noise_region[i,j]):
                # distance from center
                ang = rotation_map_copy[i,j]
                if ang >= 360.:
                    ang -= 360
                val = noise_region[i,j]
                angs.append(ang)
                vals.append(val)
    
    angs = np.array(angs)
    
    coeffs = np.polyfit(angs, vals, 4)
    #x_arr = np.arange(np.min(angs), np.max(angs), 0.01)
    #y_fit = coeffs[4] + coeffs[3]*x_arr + coeffs[2]*x_arr**2 + coeffs[1]*x_arr**3 + coeffs[0]*x_arr**4 
    
    az_corrected_region = np.copy(noise_region)
    

    for i in range(imsz):
        for j in range(imsz):
            if ~np.isnan(noise_region[i,j]):
                # distance from center
                ang = rotation_map_copy[i,j]
                if ang >= 360.:
                    ang -= 360
                    
                az_fit = coeffs[4] + coeffs[3]*ang + coeffs[2]*ang**2 + coeffs[1]*ang**3 + coeffs[0]*ang**4 
                az_corrected_region[i,j] -= az_fit
                
    return az_corrected_region

def get_planet_locations_and_info(roll_angle, planet_pos_mas, pix_radius, im_dir_path):
    # open an image just to get some information about it
    
    test_dir = im_dir_path + "scatteredlight-Mp_1.0-ap_1.0-incl_00-longitude_00-exozodis_1-distance_10-rang_{}/".format(round(roll_angle))
    
    sci_im_fits = pyfits.open(test_dir + "/DET/sci_imgs.fits")
    sci_im = sci_im_fits[0].data[0, 0]
    imsc = sci_im_fits[0].header["IMSC"] # lam/D
    imsz = sci_im_fits[0].header["IMSZ"] # pix
    #wave = sci_im_fits["WAVE"].data[0]
    diam = sci_im_fits[0].header["DIAM"]


    central_pixel = (imsz - 1)/2
    loc_of_planet_pix = planet_pos_mas/imsc

    #sci_x = loc_of_planet_pix
    #sci_y = 0 

    sci_signal_i = round(central_pixel)
    sci_signal_j = round(central_pixel + loc_of_planet_pix)


    aperture = get_psf_stamp(np.copy(sci_im), sci_signal_i, sci_signal_j, pix_radius) > 0


    ref_x = loc_of_planet_pix * np.sin(np.deg2rad(roll_angle))
    ref_y = loc_of_planet_pix * np.cos(np.deg2rad(roll_angle))

    ref_signal_i = round(ref_x + central_pixel)
    ref_signal_j = round(ref_y + central_pixel)
    
    return sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, central_pixel, aperture, imsc, diam



def rotate_region(region_sci, im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, inner_r, aperture, ap_sz, roll_angle, opposite=False):
    
    region_ref = ~np.isnan(region_sci)
    region_ref = region_ref.astype(float)
    #region_ref[sci_signal_i-ap_sz:sci_signal_i+ap_sz+1, sci_signal_j-ap_sz:sci_signal_j+ap_sz+1] = 1
    
    region_ref = rotate(region_ref, -roll_angle, order=0, reshape=False)
    
    if opposite is False and inner_r is None:
        region_ref[ref_signal_i-ap_sz:ref_signal_i+ap_sz+1, ref_signal_j-ap_sz:ref_signal_j+ap_sz+1] = ~aperture
    
    zero_inds = np.where(region_ref == 0)
    region_ref[zero_inds] = np.nan
    
    region_ref = region_ref * im
    
    return region_ref
    
    


def calc_CC_SNR_ADI(cc_map, cc_map_single, noise_map, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, ref_signal_i_opp, ref_signal_j_opp, ap_sz, noise_region_name):
    
    zero_inds = np.where(cc_map == 0)
    cc_map[zero_inds] = np.nan
    
    zero_inds = np.where(cc_map_single == 0)
    cc_map_single[zero_inds] = np.nan
    
    cc_sig = cc_map[sci_signal_i, sci_signal_j]
    
    
    
    
    noise_map[sci_signal_i-2*ap_sz:sci_signal_i+2*ap_sz+1, sci_signal_j-2*ap_sz:sci_signal_j+2*ap_sz+1] = 0
    if noise_region_name == "ring":
        noise_map[ref_signal_i-2*ap_sz:ref_signal_i+2*ap_sz+1, ref_signal_j-2*ap_sz:ref_signal_j+2*ap_sz+1] = 1
    else:
        noise_map[ref_signal_i-2*ap_sz:ref_signal_i+2*ap_sz+1, ref_signal_j-2*ap_sz:ref_signal_j+2*ap_sz+1] = 0
        noise_map[ref_signal_i_opp-2*ap_sz:ref_signal_i_opp+2*ap_sz+1, ref_signal_j_opp-2*ap_sz:ref_signal_j_opp+2*ap_sz+1] = 0

    
    cc_noise_map = cc_map_single*noise_map
    zero_inds = np.where(cc_noise_map == 0)
    cc_noise_map[zero_inds] = np.nan
    
    
    
    cc_noise_vals = cc_noise_map[~np.isnan(cc_noise_map)]
        
    cc_noise = np.sqrt(np.nanstd(cc_noise_vals, ddof=1)**2 + cc_sig)

    cc_SNR = np.abs(cc_sig - np.median(cc_noise_vals)) / cc_noise

    
    return cc_SNR

def calc_CC_SNR_RDI(cc_map, noise_map, sci_signal_i, sci_signal_j, ap_sz, noise_region_name):
    
    zero_inds = np.where(cc_map == 0)
    cc_map[zero_inds] = np.nan
    cc_sig = cc_map[sci_signal_i, sci_signal_j]
    
    
    noise_map[sci_signal_i-2*ap_sz:sci_signal_i+2*ap_sz+1, sci_signal_j-2*ap_sz:sci_signal_j+2*ap_sz+1] = 0
    
    
    cc_noise_map = cc_map*noise_map
    zero_inds = np.where(cc_noise_map == 0)
    cc_noise_map[zero_inds] = np.nan
    
    
    cc_noise_vals = cc_noise_map[~np.isnan(cc_noise_map)]
    
    cc_noise = np.sqrt(np.nanstd(cc_noise_vals, ddof=1)**2 + cc_sig)
    
    cc_SNR = np.abs(cc_sig) / cc_noise
    
    return cc_SNR



def calc_SNR_ttest(signal_apertures, noise_apertures, DI):
    
    # t-test:
    # SNR = (x1 - x2)/(s2*sqrt(1+1/n2))
    # where:
    # x1 = intensity of signal apertures
    # x2 = average intensity of remaining noise apertures
    # n1 = number of signal apertures 
    # n2 = number of remaining noise apertures
    # s12 = combined standard deviation of noise apertures

    n1 = 1
    n2 = len(noise_apertures)
    x1 = np.abs(signal_apertures)
    x2 = np.mean(noise_apertures)
    
    
    term1 = np.sum((x1-np.mean(x1))**2)
    term2 = np.sum((noise_apertures-np.mean(noise_apertures))**2)
    
    s12 = np.sqrt( (term1 + term2) / (n1+n2-2) )

    #print("x1", x1)
    #print("x2", x2)
    #print("x1-x2", (x1 - x2))
    #print("ttest denom:", (s12*np.sqrt(1/n1 + 1/n2)))
    SNR_ttest = (x1 - x2) / (s12*np.sqrt(1/n1 + 1/n2))

    
    
    if DI == "ADI":
        signal_ttest = (x1 - x2)
    elif DI == "RDI":
        signal_ttest = np.abs(x1 - np.abs(x2))
    noise_ttest = (s12*np.sqrt(1/n1 + 1/n2))
    
    return signal_ttest, noise_ttest
    
    




def calc_SNR_ttest_ADI(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                       aperture, ap_sz, width, height, roll_angle, corrections=False, verbose=False, out=False, noise_region=None):
    
    

    
    imsz, imsz = im.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    imctr = (imsz -1)/2
    
    
    sci_signal_i_opp, sci_signal_j_opp  = get_opp_coords(sci_signal_i, sci_signal_j, imctr)
    ref_signal_i_opp, ref_signal_j_opp  = get_opp_coords(ref_signal_i, ref_signal_j, imctr)
    
    
    sci_signal_mask = np.zeros_like(im, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    ref_signal_mask = np.zeros_like(im, dtype=bool)
    ref_signal_mask[ref_signal_i-ap_rad:ref_signal_i+ap_rad+1, ref_signal_j-ap_rad: ref_signal_j+ap_rad+1] = aperture
    
    sci_sig = im[sci_signal_mask]
    ref_sig = im[ref_signal_mask]
    
    signal_apertures = np.sum(sci_sig) + -1*np.sum(ref_sig)
    
    
    
    
    ## define noise region
# =============================================================================
#     nr_dynasquare_sci = region_dynasquare(im, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, width, height, opposite=True)
#     nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, im, sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, aperture, ap_sz, roll_angle, opposite=True)
#     
# =============================================================================
    nr_dynasquare_sci = region(im, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, inner_r, outer_r, opposite=True)
    nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, im, sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, aperture, ap_sz, roll_angle, opposite=True)

    
# =============================================================================
#     nr_dynasquare_sci = region_dynasquare(im, sci_signal_i, sci_signal_j, aperture, ap_sz, width, height, opposite=False)
#     nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz, roll_angle)
#     
# =============================================================================
# =============================================================================
#     plot_im(nr_dynasquare_sci, sci_signal_i_opp, sci_signal_j_opp)
#     plot_im(nr_dynasquare_ref, ref_signal_i_opp, ref_signal_j_opp)
#     assert False
# =============================================================================
    
    
    if corrections:
        #### do an r^2 correction on the region
        nr_dynasquare_sci = r2_correction(nr_dynasquare_sci)
        nr_dynasquare_ref = r2_correction(nr_dynasquare_ref)
        
# =============================================================================
#         #### do an azimuthal correction on the wedge  region
#         nr_wedge_sci = az_correction(nr_wedge_sci, rotation_map)
#         nr_wedge_sci_opp = az_correction(nr_wedge_sci_opp, rotation_map)
#         nr_wedge_ref = az_correction(nr_wedge_ref, rotation_map)
#         nr_wedge_ref_opp = az_correction(nr_wedge_ref_opp, rotation_map)
# =============================================================================


    ## measure noise
    counts_per_ap_nr_dynasquare_sci, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    counts_per_ap_nr_dynasquare_ref, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, ap_sz)
    
    ap_coords = np.concatenate((ap_coords_nr_dynasquare_sci, ap_coords_nr_dynasquare_ref))
    tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci# - np.median(counts_per_ap_nr_dynasquare_sci)
    tot_ref_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_ref# - np.median(counts_per_ap_nr_dynasquare_ref)
    
    
    
    ##### check if sci and ref regions have equal number of apertures sampled
# =============================================================================
#     if len(tot_sci_ap_counts_dynasquare) == len(tot_ref_ap_counts_dynasquare):
#         pass
#     elif len(tot_sci_ap_counts_dynasquare) > len(tot_ref_ap_counts_dynasquare):
#         num_inds_to_cut = len(tot_sci_ap_counts_dynasquare) - len(tot_ref_ap_counts_dynasquare)
#         tot_sci_ap_counts_dynasquare = tot_sci_ap_counts_dynasquare[:-num_inds_to_cut]
#     elif len(tot_sci_ap_counts_dynasquare) < len(tot_ref_ap_counts_dynasquare):
#         num_inds_to_cut = len(tot_ref_ap_counts_dynasquare) - len(tot_sci_ap_counts_dynasquare)
#         tot_ref_ap_counts_dynasquare = tot_ref_ap_counts_dynasquare[:-num_inds_to_cut]
#     else:
#         assert False, "Something is wrong"
# =============================================================================
        
    
    
    
    #noise_apertures = tot_sci_ap_counts_dynasquare + -1 * tot_ref_ap_counts_dynasquare
    noise_apertures = np.concatenate((tot_sci_ap_counts_dynasquare, tot_ref_ap_counts_dynasquare))

    #print("Classic SNR:", np.abs((signal_apertures -np.mean(noise_apertures)))  / np.std(noise_apertures, ddof=1))
    SNR_classic = signal_apertures  / np.sqrt(np.std(noise_apertures, ddof=1)**2 + (signal_apertures))
    #noise_apertures = np.concatenate((tot_sci_ap_counts_dynasquare, -1*tot_ref_ap_counts_dynasquare))
    std_correction_factor = np.sqrt(1 + (1/len(noise_apertures)))
    measured_noise = np.nanstd(noise_apertures, ddof=1) * std_correction_factor
    
    #print((signal_apertures - np.mean(noise_apertures)) / np.sqrt(measured_noise**2 + signal_apertures))
    
    signal_ttest, noise_ttest = calc_SNR_ttest(signal_apertures, noise_apertures, "ADI")
# =============================================================================
#     print("noise ttest", noise_ttest)
#     print("signal_apertures", signal_apertures)
# =============================================================================

    total_noise = np.sqrt(noise_ttest**2 + signal_apertures)
# =============================================================================
#     print("total_noise", total_noise)
#     print("signal_apertures / total_noise", signal_apertures / total_noise)
# =============================================================================
    
    if out is True:
        total_noise = noise_ttest
    
    SNR_total = signal_ttest / total_noise
    #SNR_total = signal_apertures / total_noise
# =============================================================================
#     print("SNR_total", SNR_total)
#     assert False
# =============================================================================

    noise_map_sci = ~np.isnan(nr_dynasquare_sci) 
    
    return SNR_total, SNR_classic, signal_ttest, total_noise, noise_ttest, noise_map_sci, ap_coords

def calc_SNR_ttest_RDI(im, im_hipass, sci_signal_i, sci_signal_j, sci_signal_i_opp, sci_signal_j_opp, 
                       aperture, ap_sz, width, height, roll_angle, corrections=True, verbose=False):
    
    imsz, imsz = im_hipass.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    sci_signal_mask = np.zeros_like(im_hipass, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    
    sci_signal_mask_opp = np.zeros_like(im_hipass, dtype=bool)
    sci_signal_mask_opp[sci_signal_i_opp-ap_rad:sci_signal_i_opp+ap_rad+1, sci_signal_j_opp-ap_rad: sci_signal_j_opp+ap_rad+1] = aperture

    
    sci_sig = im[sci_signal_mask]
    sci_sig_opp = im[sci_signal_mask_opp]
    
    signal_apertures = np.sum(sci_sig) 
    signal_apertures_opp = np.sum(sci_sig_opp) 

    
# =============================================================================
#     print(signal_apertures)
#     assert False
# =============================================================================
    
    
    ## define noise region
    nr_dynasquare_sci = region_dynasquare(im_hipass, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, width, height, opposite=True)
    nr_dynasquare_sci_im = region_dynasquare(im, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, width, height, opposite=True)
    
    if corrections:
        #### do an r^2 correction on the region
        nr_dynasquare_sci = r2_correction(nr_dynasquare_sci)
    
    ## measure noise
    counts_per_ap_nr_dynasquare_sci, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    counts_per_ap_nr_dynasquare_sci_im, ap_coords_nr_dynasquare_sci_im = sum_apertures_in_region(nr_dynasquare_sci_im, aperture, ap_sz)
        
    tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci
    
    
    noise_apertures = tot_sci_ap_counts_dynasquare 
    
    std_correction_factor = np.sqrt(1 + (1/len(noise_apertures)))
    measured_noise = np.nanstd(noise_apertures, ddof=1) * std_correction_factor
    
    SNR_classic = np.abs((signal_apertures -np.mean(noise_apertures)))  / np.sqrt(np.std(noise_apertures, ddof=1)**2 + (signal_apertures -np.mean(noise_apertures)))

    
    signal_ttest, noise_ttest = calc_SNR_ttest(signal_apertures, noise_apertures, "RDI")
    
    total_noise = np.sqrt(noise_ttest**2 + signal_apertures)

    print(signal_apertures, signal_apertures_opp, np.mean(counts_per_ap_nr_dynasquare_sci_im), noise_ttest**2)
    print((signal_apertures - signal_apertures_opp) / np.sqrt(noise_ttest**2 + signal_apertures - signal_apertures_opp) )
    signal_ttest = signal_apertures - np.mean(counts_per_ap_nr_dynasquare_sci_im)
    
    SNR_total = signal_ttest / total_noise

    
    noise_map_sci = ~np.isnan(nr_dynasquare_sci) 
    
    return SNR_total, SNR_classic, signal_ttest, total_noise, noise_map_sci


def calc_SNR_HPMF_ADI(im_hipass, matched_filter_datacube, matched_filter_datacube_single,
                      sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                      aperture, ap_sz, inner_r, outer_r, roll_angle, noise_region="planet"):
    

    imsz, imsz = im_hipass.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    imctr = (imsz -1)/2
    
    
    sci_signal_i_opp, sci_signal_j_opp  = get_opp_coords(sci_signal_i, sci_signal_j, imctr)
    ref_signal_i_opp, ref_signal_j_opp  = get_opp_coords(ref_signal_i, ref_signal_j, imctr)
        
    
    
    ## define noise region
    if noise_region == "opposite":
        nr_dynasquare_sci = region(im_hipass, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, inner_r, outer_r, opposite=True)
        nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, im_hipass, sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, inner_r, aperture, ap_sz, roll_angle, opposite=True)
    elif noise_region == "planet":
        nr_dynasquare_sci = region(im_hipass, sci_signal_i, sci_signal_j, aperture, ap_sz, inner_r, outer_r, opposite=False)
        nr_dynasquare_ref = region(im_hipass, ref_signal_i, ref_signal_j, aperture, ap_sz, inner_r, outer_r, opposite=False)
    else:
        assert False, "Please specify noise region to use"
# =============================================================================
#         
#     plot_im(nr_dynasquare_sci, sci_signal_i, sci_signal_j)
#     plot_im(nr_dynasquare_ref, ref_signal_i, ref_signal_j)
#     assert False
# 
# =============================================================================
    ## measure noise
    counts_per_ap_nr_dynasquare_sci_photometry, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    counts_per_ap_nr_dynasquare_ref_photometry, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, ap_sz)
    
    ap_coords = np.concatenate((ap_coords_nr_dynasquare_sci, ap_coords_nr_dynasquare_ref))
    
  
    
    # get matched filter signal
    mf_signal = np.nansum(matched_filter_datacube[sci_signal_i, sci_signal_j] * im_hipass) / np.nansum(matched_filter_datacube[sci_signal_i, sci_signal_j]**2)
    
    # get matched filter noise
    mf_noises = []
    for ap_coord in ap_coords:
        i, j = ap_coord
        mf_single = np.nansum(matched_filter_datacube_single[i, j] * im_hipass) / np.nansum(matched_filter_datacube_single[i, j]**2)
        mf_noises.append(mf_single)
    

  
    mf_background = (np.nanstd(mf_noises, ddof=1) * np.sqrt(1 + 1/len(ap_coords)))
    
# =============================================================================
#     mf_noises_sci = []
#     for ap_coord in ap_coords_nr_dynasquare_sci:
#         i, j = ap_coord
#         mf_single = np.nansum(matched_filter_datacube_single[i, j] * im_hipass) / np.nansum(matched_filter_datacube_single[i, j]**2)
#         mf_noises_sci.append(mf_single)
#         
#     mf_noises_ref = []
#     for ap_coord in ap_coords_nr_dynasquare_ref:
#         i, j = ap_coord
#         mf_single = np.nansum(matched_filter_datacube_single[i, j] * im_hipass) / np.nansum(matched_filter_datacube_single[i, j]**2)
#         mf_noises_ref.append(mf_single)
#         
# 
#     mf_background = np.sqrt((np.nanstd(mf_noises_ref, ddof=1) * np.sqrt(1 + 1/len(ap_coords_nr_dynasquare_sci)))**2 + (np.nanstd(mf_noises_ref, ddof=1) * np.sqrt(1 + 1/len(ap_coords_nr_dynasquare_ref)))**2)
# =============================================================================
    
    mf_noise = np.sqrt(mf_background**2 + mf_signal)
    SNR_HPMF = mf_signal / mf_noise
    #print(mf_signal, mf_noise, mf_background, SNR_HPMF)

    
    return SNR_HPMF, mf_signal, mf_noise, mf_background

def calc_SNR_HPAP_ADI(im_hipass,
                      sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                      aperture, ap_sz, inner_r, outer_r, roll_angle, noise_region="planet"):
    

    imsz, imsz = im_hipass.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    imctr = (imsz -1)/2
    
    sci_signal_mask = np.zeros_like(im_hipass, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    ref_signal_mask = np.zeros_like(im_hipass, dtype=bool)
    ref_signal_mask[ref_signal_i-ap_rad:ref_signal_i+ap_rad+1, ref_signal_j-ap_rad: ref_signal_j+ap_rad+1] = aperture
    
    sci_sig = im_hipass[sci_signal_mask]
    ref_sig = im_hipass[ref_signal_mask]
    
    ap_signal = np.sum(sci_sig) + -1*np.sum(ref_sig)
    
    
    nr_dynasquare_sci = region(im_hipass, sci_signal_i, sci_signal_j, aperture, ap_sz, inner_r, outer_r, opposite=False)
    nr_dynasquare_ref = region(im_hipass, ref_signal_i, ref_signal_j, aperture, ap_sz, inner_r, outer_r, opposite=False)
    
    ## measure noise
    counts_per_ap_nr_dynasquare_sci_photometry, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    counts_per_ap_nr_dynasquare_ref_photometry, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, ap_sz)
    
    ap_coords = np.concatenate((ap_coords_nr_dynasquare_sci, ap_coords_nr_dynasquare_ref))
    
    noise_apertures = np.concatenate((counts_per_ap_nr_dynasquare_sci_photometry, counts_per_ap_nr_dynasquare_ref_photometry))  
  
    ap_background = (np.nanstd(noise_apertures, ddof=1) * np.sqrt(1 + 1/len(ap_coords)))
    
    ap_noise = np.sqrt(ap_background**2 + ap_signal)

    SNR_HPAP = ap_signal / ap_noise
    
    return SNR_HPAP, ap_signal, ap_noise, ap_background

def calc_SNR_HPMF_RDI(im_hipass, matched_filter_datacube_single,
                      sci_signal_i, sci_signal_j, 
                      aperture, ap_sz, inner_r, outer_r, noise_region=None):
    

    imsz, imsz = im_hipass.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    imctr = (imsz -1)/2
    
    
    sci_signal_i_opp, sci_signal_j_opp  = get_opp_coords(sci_signal_i, sci_signal_j, imctr)
        
    
    ## define noise region
    if noise_region == "opposite":
        nr_dynasquare_sci = region(im_hipass, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, inner_r, outer_r, opposite=True)
    elif noise_region == "planet":
        nr_dynasquare_sci = region(im_hipass, sci_signal_i, sci_signal_j, aperture, ap_sz, inner_r, outer_r, opposite=False)
    else:
        assert False, "Please specify noise region to use"


# =============================================================================
#     plot_im(nr_dynasquare_sci, sci_signal_i, sci_signal_j)
#     assert False
# =============================================================================
    
    ## measure noise
    counts_per_ap_nr_dynasquare_sci_photometry, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    
    ap_coords = ap_coords_nr_dynasquare_sci

    
    # get matched filter signal
    mf_signal = np.nansum(matched_filter_datacube_single[sci_signal_i, sci_signal_j] * im_hipass) / np.nansum(matched_filter_datacube_single[sci_signal_i, sci_signal_j]**2)
    
    # get matched filter noise
    mf_noises = []
    for ap_coord in ap_coords:
        i, j = ap_coord
        mf_single = np.nansum(matched_filter_datacube_single[i, j] * im_hipass) / np.nansum(matched_filter_datacube_single[i, j]**2)
        mf_noises.append(mf_single)
        
    mf_background = (np.nanstd(mf_noises, ddof=1) * np.sqrt(1 + 1/len(ap_coords)))

    mf_noise = np.sqrt(mf_background**2 + mf_signal)

    SNR_HPMF = mf_signal / mf_noise
    
    return SNR_HPMF, mf_signal, mf_noise, mf_background

def calc_SNR_HPAP_RDI(im_hipass,
                      sci_signal_i, sci_signal_j, 
                      aperture, ap_sz, inner_r, outer_r, noise_region=None):
    

    imsz, imsz = im_hipass.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    imctr = (imsz -1)/2
    
    sci_signal_mask = np.zeros_like(im_hipass, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    
    sci_sig = im_hipass[sci_signal_mask]
    
    ap_signal = np.sum(sci_sig) 
    
    nr_dynasquare_sci = region(im_hipass, sci_signal_i, sci_signal_j, aperture, ap_sz, inner_r, outer_r, opposite=False)
    
    
    ## measure noise
    counts_per_ap_nr_dynasquare_sci_photometry, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    
    ap_coords = ap_coords_nr_dynasquare_sci

            
    ap_background = (np.nanstd(counts_per_ap_nr_dynasquare_sci_photometry, ddof=1) * np.sqrt(1 + 1/len(ap_coords)))


    ap_noise = np.sqrt(ap_background**2 + ap_signal)

    SNR_HPAP = ap_signal / ap_noise
    
    return SNR_HPAP, ap_signal, ap_noise, ap_background

def get_optimal_filtersize(temp_bool, df, mode=None):
    if mode == "HPAP":
        noise_meas_expt_arr = df[temp_bool]["med_meas_noise_HPAP"].values / df[temp_bool]["expected_noise_bkgr_ap"].values
    elif mode == "HPMF":
        noise_meas_expt_arr = df[temp_bool]["med_meas_noise_HPMF"].values / df[temp_bool]["expected_noise_bkgr_mf"].values
    else:
        assert False, "Check mode, must be HPAP or HPMF"
    # limit to filter size less than 20 pixels
    filter_size_arr_temp = np.copy(df[temp_bool]["filter_sz_pix"].values)
    inds_gt_one = (np.abs(noise_meas_expt_arr) > 0.95) & (filter_size_arr_temp > 1.)
    
    optimal_ind = np.argmin(np.abs(noise_meas_expt_arr[inds_gt_one] - 1.))
    optimal_filtersz = filter_size_arr_temp[inds_gt_one][optimal_ind]
    
    return optimal_filtersz
    
    
