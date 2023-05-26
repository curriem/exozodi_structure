import numpy as np
import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, shift, rotate
from scipy.interpolate import NearestNDInterpolator

def plot_im(im, signal_i, signal_j, log=False):
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



def synthesize_images_ADI(im_dir, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, zodis, aperture,
                          target_SNR=7, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, r2_disk=False):
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

    
    
    
#     plot_im(aperture_mask, plan_i, plan_j)
    
    sci_plan_CR = np.sum(sci_plan_im * sci_aperture_mask)
    ref_plan_CR = np.sum(ref_plan_im * ref_aperture_mask)
    
    
    if add_star:
        sci_star_CR1 = np.sum(sci_star_im * sci_aperture_mask)
        sci_star_CR2 = np.sum(ref_star_im * sci_aperture_mask)

        ref_star_CR1 = np.sum(ref_star_im * ref_aperture_mask)
        ref_star_CR2 = np.sum(sci_star_im * ref_aperture_mask)

        
        sci_star_CR1_out = np.sum(sci_star_im * sci_aperture_mask_out)
        sci_star_CR2_out = np.sum(ref_star_im * sci_aperture_mask_out)
        ref_star_CR1_out = np.sum(ref_star_im * ref_aperture_mask_out)
        ref_star_CR2_out = np.sum(sci_star_im * ref_aperture_mask_out)

    else:
        sci_star_CR1 = 0
        ref_star_CR1 = 0
        sci_star_CR2 = 0
        ref_star_CR2 = 0
        
        sci_star_CR1_out = 0
        ref_star_CR1_out = 0
        sci_star_CR2_out = 0
        ref_star_CR2_out = 0
        
        sci_star_im = np.zeros_like(sci_star_im)
        ref_star_im = np.zeros_like(ref_star_im)
        
  
        
    sci_disk_im_fits = pyfits.open(im_dir + "/DET/sci_disk.fits")
    sci_disk_im = sci_disk_im_fits[0].data[0, 0]
    
    ref_disk_im_fits = pyfits.open(im_dir + "/DET/ref_disk.fits")
    ref_disk_im = ref_disk_im_fits[0].data[0, 0]
    

    if uniform_disk:
        
        if add_star:
            disk_val_at_planet = sci_disk_im[sci_plan_i, sci_plan_j]
        else:
            disk_val_at_planet = sci_disk_im[sci_plan_i, sci_plan_j]*100
        sci_disk_im = disk_val_at_planet * np.ones_like(sci_disk_im)
        ref_disk_im = disk_val_at_planet * np.ones_like(ref_disk_im)
        
    if r2_disk:
        center_disk_val = 10.
        r2_disk = np.empty_like(sci_disk_im)
        center = 50
        for i in range(101):
            for j in range(101):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                val = center_disk_val /dist**2
                r2_disk[i, j] = val
        r2_disk[50, 50] = center_disk_val
        
        sci_disk_im = r2_disk
        ref_disk_im = r2_disk
        
    
    
    #sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
    #ref_disk_CR = np.sum(ref_disk_im*ref_aperture_mask)
    
    sci_disk_CR1 = np.sum(sci_disk_im*sci_aperture_mask)
    sci_disk_CR2 = np.sum(ref_disk_im*sci_aperture_mask)

    ref_disk_CR1 = np.sum(ref_disk_im*ref_aperture_mask)
    ref_disk_CR2 = np.sum(sci_disk_im*ref_aperture_mask)

    
    

    
    sci_back_CR1 = sci_disk_CR1
    sci_back_CR2 = sci_disk_CR2
    ref_back_CR1 = ref_disk_CR1
    ref_back_CR2 = ref_disk_CR2
    
    
    ######### outside region ###########
    sci_disk_CR1_out = np.sum(sci_disk_im*sci_aperture_mask_out)
    sci_disk_CR2_out = np.sum(ref_disk_im*sci_aperture_mask_out)

    ref_disk_CR1_out = np.sum(ref_disk_im*ref_aperture_mask_out)
    ref_disk_CR2_out = np.sum(sci_disk_im*ref_aperture_mask_out)

    
    sci_back_CR1_out = sci_disk_CR1_out
    ref_back_CR1_out = ref_disk_CR1_out
    sci_back_CR2_out = sci_disk_CR2_out
    ref_back_CR2_out = ref_disk_CR2_out
    
    sci_back_CR1 += sci_star_CR1
    ref_back_CR1 += ref_star_CR1
    sci_back_CR2 += sci_star_CR2
    ref_back_CR2 += ref_star_CR2
    
    sci_back_CR1_out += sci_star_CR1_out
    ref_back_CR1_out += ref_star_CR1_out
    sci_back_CR2_out += sci_star_CR2_out
    ref_back_CR2_out += ref_star_CR2_out
    ###################################################
    
    
    
    # tot_noise_CR = 2.*sci_back_CR # ph/s
    tot_noise_CR = sci_back_CR1 + ref_back_CR1 #  + sci_back_CR2 + ref_back_CR2 # ph/s
    tot_noise_CR_out = sci_back_CR1_out + ref_back_CR1_out # + sci_back_CR2_out + ref_back_CR2_out # ph/s
    
    
    tot_background_CR = np.copy(tot_noise_CR)
    tot_background_CR_out = np.copy(tot_noise_CR_out)
    
    
    #if planet_noise:
    #    tot_noise_CR += sci_plan_CR
    #    tot_noise_CR += ref_plan_CR
    
    #tot_tint = target_SNR**2 * tot_noise_CR/(sci_plan_CR+ref_plan_CR)**2 # s

    sci_sig_sources = (sci_plan_CR + sci_disk_CR1 + sci_star_CR1 - sci_disk_CR2 - sci_star_CR2) 
    ref_sig_sources = (ref_plan_CR + ref_disk_CR1 + ref_star_CR1 - ref_disk_CR2 - ref_star_CR2) 
    sci_noise_sources = (sci_plan_CR + sci_disk_CR1 + sci_star_CR1 + sci_disk_CR2 + sci_star_CR2) 
    ref_noise_sources = (ref_plan_CR + ref_disk_CR1 + ref_star_CR1 + ref_disk_CR2 + ref_star_CR2) 

    all_signal_sources = sci_sig_sources + ref_sig_sources
    all_noise_sources = sci_noise_sources + ref_noise_sources
        

    tot_tint = target_SNR**2 * all_noise_sources/(all_signal_sources)**2 # s
    
    sci_tint = tot_tint / 2
    ref_tint = tot_tint/2
    
    if verbose:
# =============================================================================
#         print("Sci planet counts:", sci_plan_CR*tot_tint)
#         print("Sci Disk counts:", (sci_disk_CR1 )*tot_tint)
#         print("Sci Star counts:", (sci_star_CR1 )*tot_tint)
#         print("Sci Integration time:", tot_tint)
#         
#         print("Ref planet counts:", ref_plan_CR*tot_tint)
#         print("Ref Disk counts:", (ref_disk_CR1 )*tot_tint)
#         print("Ref Star counts:", (ref_star_CR1 )*tot_tint)
# =============================================================================
# =============================================================================
#         print("Ref Integration time:", tot_tint)
# =============================================================================
        
        

        print("All sources at sci loc in sub im", sci_sig_sources * tot_tint)
        print("All sources at ref loc in sub im", ref_sig_sources * tot_tint)
        
        print("All signal sources:", all_signal_sources * tot_tint)
        print("All noise sources:", all_noise_sources * tot_tint)

        
        print("tot_tint", tot_tint)
        SNR_calc = all_signal_sources * tot_tint / np.sqrt(all_noise_sources*tot_tint)
        print("SNR_calc:", SNR_calc)

        
        
# =============================================================================
#         if planet_noise:
# # =============================================================================
# #             SNR_calc = (sci_plan_CR*tot_tint +ref_plan_CR*tot_tint) / np.sqrt((sci_disk_CR1 + sci_disk_CR2 + 
# #                                                                                ref_disk_CR1 + ref_disk_CR2 + 
# #                                                                                sci_star_CR1 + sci_star_CR2 + 
# #                                                                                ref_star_CR1 + ref_star_CR2 + 
# #                                                                                sci_plan_CR + ref_plan_CR)*tot_tint) 
# # =============================================================================
#             
#             SNR_calc = (sci_plan_CR*tot_tint +ref_plan_CR*tot_tint) / np.sqrt((sci_disk_CR1 + 
#                                                                                ref_disk_CR1 + 
#                                                                                sci_star_CR1 + 
#                                                                                ref_star_CR1 + 
#                                                                                sci_plan_CR + ref_plan_CR)*tot_tint) 
#             
#         else:
#             SNR_calc = (sci_plan_CR*tot_tint +ref_plan_CR*tot_tint) / np.sqrt((sci_disk_CR1 + sci_disk_CR2 + 
#                                                                                ref_disk_CR1 + ref_disk_CR2 + 
#                                                                                sci_star_CR1 + sci_star_CR2 + 
#                                                                                ref_star_CR1 + ref_star_CR2)*tot_tint) 
# =============================================================================
        print("SNR calculation:", SNR_calc)
        
    sci_planet_counts, ref_planet_counts = sci_plan_CR*tot_tint, ref_plan_CR*tot_tint
    
    science_image = (sci_plan_im + sci_disk_im + sci_star_im) * sci_tint
    reference_image = (ref_plan_im + ref_disk_im + ref_star_im) * ref_tint
    
    if planet_noise:
        science_poisson = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * sci_tint)
        reference_poisson = np.random.poisson((ref_disk_im + ref_star_im + ref_plan_im) * ref_tint)
        
    else:
        science_poisson = np.random.poisson((sci_disk_im + sci_star_im) * sci_tint)
        reference_poisson = np.random.poisson((ref_disk_im + ref_star_im) * ref_tint)
    
    science_poisson_noplan = np.random.poisson((sci_disk_im + sci_star_im) * sci_tint)
    reference_poisson_noplan = np.random.poisson((ref_disk_im + ref_star_im) * ref_tint)

# =============================================================================
#     plt.figure()
#     plt.imshow(np.log10(reference_poisson))
#     plt.colorbar()
#     plt.show()
#     assert False
# =============================================================================
    
    
    if add_noise:
        science_image += science_poisson
        reference_image += reference_poisson
        
        science_image_noplan =  (sci_disk_im + sci_star_im) * tot_tint + science_poisson_noplan
        reference_image_noplan = (ref_disk_im + ref_star_im) * tot_tint + reference_poisson_noplan
        
    
    
    
    tot_noise_counts = tot_background_CR*tot_tint
    tot_noise_counts_out = tot_background_CR_out*tot_tint
    

    return science_image, reference_image, sci_planet_counts, ref_planet_counts, tot_noise_counts, tot_noise_counts_out, (sci_out_i, sci_out_j, ref_out_i, ref_out_j)

def synthesize_images_ADI2(im_dir, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, zodis, aperture,
                          target_SNR=7, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, simple_planet=False):
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
    
    if simple_planet:
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
    

    if uniform_disk:
        
        disk_val_at_planet = sci_disk_im[sci_plan_i, sci_plan_j]
        sci_disk_im = disk_val_at_planet * np.ones_like(sci_disk_im)
        ref_disk_im = disk_val_at_planet * np.ones_like(ref_disk_im)
        
    
    if add_star:
        pass
    else:
        sci_star_im = np.zeros_like(sci_star_im)
        ref_star_im = np.zeros_like(ref_star_im)
        
    
        
    
    
    
    sci_star_CR = np.sum(sci_star_im * sci_aperture_mask)
    ref_star_CR = np.sum(ref_star_im * ref_aperture_mask)
    
    sci_star_CR_out = np.sum(sci_star_im * sci_aperture_mask_out)
    ref_star_CR_out = np.sum(ref_star_im * ref_aperture_mask_out)
    
    
    
    sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
    ref_disk_CR = np.sum(ref_disk_im*ref_aperture_mask)

    sci_disk_CR_out = np.sum(sci_disk_im*sci_aperture_mask_out)
    ref_disk_CR_out = np.sum(ref_disk_im*ref_aperture_mask_out)
    


    sci_bkgr_CR = sci_disk_CR + sci_star_CR
    ref_bkgr_CR = ref_disk_CR + ref_star_CR
        
    sci_bkgr_CR_out = sci_disk_CR_out + sci_star_CR_out
    ref_bkgr_CR_out = ref_disk_CR_out + ref_star_CR_out
    
    
    tot_bkgr_CR = sci_bkgr_CR + ref_bkgr_CR
    tot_bkgr_CR_out = sci_bkgr_CR_out + ref_bkgr_CR_out


    total_signal_CR = sci_plan_CR + ref_plan_CR
    total_noise_CR = total_signal_CR + tot_bkgr_CR
    
        
    
    tot_tint = target_SNR**2 * total_noise_CR/(total_signal_CR)**2 # s
    
    
    
    tot_noise_counts = total_noise_CR*tot_tint
    tot_noise_counts_out = tot_bkgr_CR_out*tot_tint
    
    if verbose:
        print("Total Signal Counts:", total_signal_CR*tot_tint)
        print("Total Background Counts:", (tot_bkgr_CR )*tot_tint)
        print("Total Noise Counts:", (total_noise_CR )*tot_tint)
 
        
        print("tot_tint", tot_tint)
        SNR_calc = total_signal_CR * tot_tint / np.sqrt(total_noise_CR*tot_tint)
        print("SNR_calc:", SNR_calc)

        
        
    sci_planet_counts, ref_planet_counts = sci_plan_CR*tot_tint, ref_plan_CR*tot_tint
    
    science_image = (sci_plan_im + sci_disk_im + sci_star_im) * tot_tint
    reference_image = (ref_plan_im + ref_disk_im + ref_star_im) * tot_tint
    
    science_image_noisy = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * tot_tint)
    reference_image_noisy = np.random.poisson((ref_disk_im + ref_star_im + ref_plan_im) * tot_tint)
    
    
    
    if add_noise:
        science_image = science_image_noisy.astype(float)
        reference_image = reference_image_noisy.astype(float)
    
    

    return science_image, reference_image, sci_planet_counts, ref_planet_counts, tot_noise_counts, tot_noise_counts_out, (sci_out_i, sci_out_j, ref_out_i, ref_out_j)


def synthesize_images_ADI3(im_dir, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, incl, zodis, aperture, roll_angle,
                          target_SNR=7, tot_tint=None, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, simple_planet=False):
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

    
# =============================================================================
#     plt.figure()
#     plt.imshow(sci_disk_im*1000)
#     plt.colorbar()
#     
#     plt.figure()
#     plt.imshow(ref_disk_im*1000)
#     plt.colorbar()
#     plt.figure()
#     plt.imshow((sci_disk_im - ref_disk_im)*1000)
#     plt.colorbar()
#     plt.title("loc1: {}, loc2: {}".format(round(np.sum(sub_disk_im*1000*sci_aperture_mask)), round(np.sum(sub_disk_im*1000*ref_aperture_mask))))
#     
#     
#     
#     
#     assert False
# =============================================================================
    
    

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
    
    
    height=3
    width=3
    
    ## define noise region
    nr_dynasquare_sci = region_dynasquare(sci_im_total, sci_signal_i_opp, sci_signal_j_opp, aperture, pix_radius, width, height, opposite=True)
    nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, ref_im_total, sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, aperture, pix_radius, roll_angle, opposite=True)
    
# =============================================================================
#     ## define noise region
#     nr_dynasquare_sci = region_dynasquare(sci_im_total, sci_plan_i, sci_plan_j, aperture, pix_radius, width, height, opposite=False)
#     nr_dynasquare_ref = rotate_fregion(nr_dynasquare_sci, ref_im_total, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, aperture, pix_radius, roll_angle)
# =============================================================================


    ## measure noise
    counts_per_ap_nr_dynasquare_sci, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, pix_radius)
    counts_per_ap_nr_dynasquare_ref, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, pix_radius)
    
        
    tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci# - np.median(counts_per_ap_nr_dynasquare_sci)
    tot_ref_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_ref# - np.median(counts_per_ap_nr_dynasquare_ref)    

    


    #signal_ttest, noise_ttest = calc_SNR_ttest(signal_apertures, noise_apertures)

    noise_background = np.mean(tot_sci_ap_counts_dynasquare) + np.mean(tot_ref_ap_counts_dynasquare)
    
    
    total_signal_CR = signal_apertures  #- noise_ttest#np.mean(noise_apertures)
    if incl == 0 and (zodis < 10):
        total_signal_CR  = total_signal_CR + np.sum(sub_disk_im*sci_aperture_mask) + -1*np.sum(sub_disk_im*ref_aperture_mask)
    total_noise_CR = noise_background + total_signal_CR
    total_bkgr_CR = noise_background
    
    if target_SNR is None and tot_tint is None:
        assert False, "target_SNR and tot_tint are both None. Set one to something."
    elif target_SNR is not None and tot_tint is not None:
        assert False, "target_SNR and tot_tint cannot both be set!"
    elif target_SNR is not None and tot_tint is None:
        tot_tint = target_SNR**2 * total_noise_CR / total_signal_CR**2
    elif target_SNR is None and tot_tint is not None:
        pass
    else:
        assert False, "Something is wrong... check target_SNR and tot_tint"
        
    

    
    # get noise estimate for outside
    
    sci_signal_i_opp_out, sci_signal_j_opp_out  = get_opp_coords(sci_out_i, sci_out_j, 50)
    ref_signal_i_opp_out, ref_signal_j_opp_out  = get_opp_coords(ref_out_i, ref_out_j, 50)
    
    ## define noise region
    nr_dynasquare_sci_out = region_dynasquare(sci_im_total, sci_signal_i_opp_out, sci_signal_j_opp_out, aperture, pix_radius, width, height, opposite=True)
    nr_dynasquare_ref_out = rotate_region(nr_dynasquare_sci, ref_im_total, sci_signal_i_opp_out, sci_signal_j_opp_out, ref_signal_i_opp_out, ref_signal_j_opp_out, aperture, pix_radius, roll_angle, opposite=True)
    
# =============================================================================
#     ## define noise region
#     nr_dynasquare_sci = region_dynasquare(sci_im_total, sci_plan_i, sci_plan_j, aperture, pix_radius, width, height, opposite=False)
#     nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, ref_im_total, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, aperture, pix_radius, roll_angle)
# =============================================================================


    ## measure noise
    counts_per_ap_nr_dynasquare_sci_out, ap_coords_nr_dynasquare_sci_out = sum_apertures_in_region(nr_dynasquare_sci_out, aperture, pix_radius)
    counts_per_ap_nr_dynasquare_ref_out, ap_coords_nr_dynasquare_ref_out = sum_apertures_in_region(nr_dynasquare_ref_out, aperture, pix_radius)
    
        
    tot_sci_ap_counts_dynasquare_out = counts_per_ap_nr_dynasquare_sci_out# - np.median(counts_per_ap_nr_dynasquare_sci)
    tot_ref_ap_counts_dynasquare_out = counts_per_ap_nr_dynasquare_ref_out# - np.median(counts_per_ap_nr_dynasquare_ref)    

    


    #signal_ttest, noise_ttest = calc_SNR_ttest(signal_apertures, noise_apertures)

    noise_background_out = np.mean(tot_sci_ap_counts_dynasquare_out) + np.mean(tot_ref_ap_counts_dynasquare_out)
    total_bkgr_CR_out = noise_background_out
    
    

    
    if verbose:
        print("signal CR:", total_signal_CR)
        print("noise CR:", total_noise_CR)

        
        print("Total Signal Counts:", total_signal_CR*tot_tint)
        print("Total Background Counts:", (total_bkgr_CR )*tot_tint)
        
        print("Total Noise Counts:", (total_noise_CR )*tot_tint)
 
        
        print("tot_tint", tot_tint)
        SNR_calc = total_signal_CR * tot_tint / np.sqrt(total_noise_CR*tot_tint)
        print("SNR_calc:", SNR_calc)
        
        
    
    
    tot_noise_counts = total_noise_CR*tot_tint

    tot_noise_counts_out = total_bkgr_CR_out*tot_tint


    science_image = (sci_plan_im + sci_disk_im + sci_star_im) * tot_tint
    reference_image = (ref_plan_im + ref_disk_im + ref_star_im) * tot_tint
    
    science_image_noisy = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * tot_tint)
    reference_image_noisy = np.random.poisson((ref_disk_im + ref_star_im + ref_plan_im) * tot_tint)
    
    
    
    if add_noise:
        science_image = science_image_noisy.astype(float)
        reference_image = reference_image_noisy.astype(float)
    
    

    return science_image, reference_image, tot_noise_counts, tot_noise_counts_out, (sci_out_i, sci_out_j, ref_out_i, ref_out_j), tot_tint


    
  
def synthesize_images_RDI(im_dir, sci_plan_i, sci_plan_j, zodis, aperture,
                          target_SNR=7, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, r2_disk=False):
    """
    

    Parameters
    ----------
    im_dir : TYPE
        DESCRIPTION.
    sci_plan_i : TYPE
        DESCRIPTION.
    sci_plan_j : TYPE
        DESCRIPTION.
    zodis : TYPE
        DESCRIPTION.
    aperture : TYPE
        DESCRIPTION.
    target_SNR : TYPE, optional
        DESCRIPTION. The default is 7.
    pix_radius : TYPE, optional
        DESCRIPTION. The default is 1.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    planet_noise : TYPE, optional
        DESCRIPTION. The default is True.
    add_noise : TYPE, optional
        DESCRIPTION. The default is True.
    add_star : TYPE, optional
        DESCRIPTION. The default is True.
    uniform_disk : TYPE, optional
        DESCRIPTION. The default is False.
    r2_disk : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    sci_plan_im_fits = pyfits.open(im_dir + "/DET/sci_plan.fits")
    sci_plan_im = sci_plan_im_fits[0].data[0, 0]
    
    sci_star_im_fits = pyfits.open(im_dir + "/DET/sci_star.fits")
    sci_star_im = sci_star_im_fits[0].data[0,0]
    
    ref_star_im_fits = pyfits.open(im_dir + "/DET/ref_star.fits")
    ref_star_im = ref_star_im_fits[0].data[0,0]
    
    
    sci_aperture_mask = np.zeros_like(sci_plan_im)
    sci_aperture_mask[sci_plan_i-pix_radius:sci_plan_i+pix_radius+1, sci_plan_j-pix_radius:sci_plan_j+pix_radius+1] = aperture
    
    
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
    
    
    
#     plot_im(aperture_mask, plan_i, plan_j)
    
    sci_plan_CR = np.sum(sci_plan_im * sci_aperture_mask)
    
    if add_star:
        sci_star_CR = np.sum(sci_star_im * sci_aperture_mask)
        ref_star_CR = np.sum(ref_star_im * sci_aperture_mask)
        
        sci_star_CR_out = np.sum(sci_star_im * sci_aperture_mask_out)
        ref_star_CR_out = np.sum(ref_star_im * sci_aperture_mask_out)
    else:
        sci_star_CR = 0
        sci_star_im = np.zeros_like(sci_star_im)
        ref_star_CR = 0
        ref_star_im = np.zeros_like(ref_star_im)
        sci_star_CR_out = 0
        ref_star_CR_out = 0
        
    sci_disk_im_fits = pyfits.open(im_dir + "/DET/sci_disk.fits")
    sci_disk_im = sci_disk_im_fits[0].data[0, 0]
    
# =============================================================================
#     ref_disk_im_fits = pyfits.open(im_dir + "/DET/ref_disk.fits")
#     ref_disk_im = ref_disk_im_fits[0].data[0, 0]
# =============================================================================
    

    if uniform_disk:
        disk_val_at_planet = sci_disk_im[sci_plan_i, sci_plan_j]
        #disk_val_at_planet = 100
        sci_disk_im = disk_val_at_planet * np.ones_like(sci_disk_im)
        #ref_disk_im = disk_val_at_planet * np.ones_like(ref_disk_im)
        
    if r2_disk:
        center_disk_val = 10.
        r2_disk = np.empty_like(sci_disk_im)
        center = 50
        for i in range(101):
            for j in range(101):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                val = center_disk_val /dist**2
                r2_disk[i, j] = val
        r2_disk[50, 50] = center_disk_val
        
        sci_disk_im = r2_disk
        #ref_disk_im = r2_disk
        
    
    
    sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
    
        
    sci_back_CR = sci_disk_CR 
    ref_back_CR = 0.
    ref_back_CR_out = 0.
    
    ######### outside region ###########
    sci_disk_CR_out = np.sum(sci_disk_im*sci_aperture_mask_out)
    
    
    sci_back_CR_out = sci_disk_CR_out
    
    if add_star:
        sci_back_CR += sci_star_CR
        ref_back_CR += ref_star_CR
        sci_back_CR_out += sci_star_CR_out
        ref_back_CR_out += ref_star_CR_out
    ###################################################
    
    # tot_noise_CR = 2.*sci_back_CR # ph/s
    tot_noise_CR = sci_back_CR + ref_back_CR # ph/s
    tot_noise_CR_out = sci_back_CR_out + ref_back_CR_out # ph/s
    
    tot_background_CR = np.copy(tot_noise_CR)
    tot_background_CR_out = np.copy(tot_noise_CR_out)
    
    if planet_noise:
        tot_noise_CR += sci_plan_CR

        
    tot_tint = target_SNR**2 * tot_noise_CR/sci_plan_CR**2 # s
    

    sci_tint = tot_tint
    ref_tint = tot_tint

    
    if verbose:
        print("Sci planet counts:", sci_plan_CR*sci_tint)
        print("Sci Disk counts:", sci_disk_CR*sci_tint)
        print("Sci Star counts:", sci_star_CR*sci_tint)
        print("Sci Integration time:", sci_tint)
        
        #print("Ref planet counts:", ref_plan_CR*ref_tint)
        #print("Ref Disk counts:", ref_disk_CR*ref_tint)
        print("Ref Star counts:", ref_star_CR*ref_tint)
        print("Ref Integration time:", ref_tint)
        
        print("SNR calculation:", (sci_plan_CR*sci_tint) / np.sqrt(sci_plan_CR*sci_tint + sci_disk_CR*sci_tint +sci_star_CR*sci_tint + ref_star_CR*ref_tint) )
        
    sci_planet_counts = sci_plan_CR*sci_tint

    science_image = (sci_plan_im + sci_disk_im + sci_star_im) * sci_tint
    reference_image = ref_star_im * ref_tint
    
    if ~planet_noise:
        science_poisson = np.random.poisson((sci_disk_im + sci_star_im) * sci_tint)
        reference_poisson = np.random.poisson(ref_star_im * ref_tint)
    elif planet_noise:
        science_poisson = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * sci_tint)
        reference_poisson = np.random.poisson(ref_star_im * ref_tint)
    
    

    if add_noise:
        science_image += science_poisson
        reference_image += reference_poisson
    
    tot_noise_counts = tot_background_CR*tot_tint
    tot_noise_counts_out = tot_background_CR_out*tot_tint

    return science_image, reference_image, sci_planet_counts, tot_noise_counts, tot_noise_counts_out, (sci_out_i, sci_out_j)

def synthesize_images_RDI2(im_dir, sci_plan_i, sci_plan_j, zodis, aperture,
                          target_SNR=7, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, r2_disk=False, simple_planet=False):
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
    
    
    if simple_planet:
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
    
        
    sci_plan_CR = np.sum(sci_plan_im * sci_aperture_mask)
    
    if add_star:
        sci_star_CR = np.sum(sci_star_im * sci_aperture_mask)
        ref_star_CR = np.sum(ref_star_im * sci_aperture_mask)
        
        sci_star_CR_out = np.sum(sci_star_im * sci_aperture_mask_out)
        ref_star_CR_out = np.sum(ref_star_im * sci_aperture_mask_out)
    else:
        sci_star_CR = 0
        sci_star_im = np.zeros_like(sci_star_im)
        ref_star_CR = 0
        ref_star_im = np.zeros_like(ref_star_im)
        sci_star_CR_out = 0
        ref_star_CR_out = 0
        
    sci_disk_im_fits = pyfits.open(im_dir + "/DET/sci_disk.fits")
    sci_disk_im = sci_disk_im_fits[0].data[0, 0]
    

    if uniform_disk:
        disk_val_at_planet = sci_disk_im[sci_plan_i, sci_plan_j]
        sci_disk_im = disk_val_at_planet * np.ones_like(sci_disk_im)
        
    
    sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
    
    
    sci_disk_CR_out = np.sum(sci_disk_im*sci_aperture_mask_out)
    
    
    sci_bkgr_CR = sci_disk_CR + sci_star_CR
    ref_bkgr_CR =  ref_star_CR
        
    sci_bkgr_CR_out = sci_disk_CR_out + sci_star_CR_out
    ref_bkgr_CR_out = ref_star_CR_out
    
    total_signal_CR = sci_plan_CR 
    total_noise_CR = sci_plan_CR + sci_bkgr_CR + ref_bkgr_CR
    
    tot_tint = target_SNR**2 * total_noise_CR/(total_signal_CR)**2 # s

    tot_bkgr_CR = sci_bkgr_CR + ref_bkgr_CR
    tot_bkgr_CR_out = sci_bkgr_CR_out + ref_bkgr_CR_out
    
    tot_noise_counts = total_noise_CR*tot_tint
    tot_noise_counts_out = tot_bkgr_CR_out*tot_tint
    
    if verbose:
        print("Total Signal Counts:", total_signal_CR*tot_tint)
        print("Total Background Counts:", tot_bkgr_CR *tot_tint)
        print("Total Noise Counts:", (total_noise_CR )*tot_tint)
 
        
        print("tot_tint", tot_tint)
        SNR_calc = total_signal_CR * tot_tint / np.sqrt(total_noise_CR*tot_tint)
        print("SNR_calc:", SNR_calc)

        
        
    sci_planet_counts = sci_plan_CR*tot_tint
    
    science_image = (sci_plan_im + sci_disk_im + sci_star_im) * tot_tint
    reference_image = (ref_star_im) * tot_tint
    
    science_image_noisy = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * tot_tint)
    reference_image_noisy = np.random.poisson((ref_star_im) * tot_tint)
    
    
    
    if add_noise:
        science_image = science_image_noisy.astype(float)
        reference_image = reference_image_noisy.astype(float)
    


    return science_image, reference_image, sci_planet_counts, tot_noise_counts, tot_noise_counts_out, (sci_out_i, sci_out_j)


def synthesize_images_RDI3(im_dir, sci_plan_i, sci_plan_j, zodis, aperture,
                          target_SNR=7, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, r2_disk=False, simple_planet=False, zerodisk=False):
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
    
    
    
    sci_sig_CR = sub_im_total * sci_aperture_mask
    
    signal_apertures = np.sum(sci_sig_CR)
    

    
    
    
    sci_signal_i_opp, sci_signal_j_opp  = get_opp_coords(sci_plan_i, sci_plan_j, (101-1)/2)
    
    
    height=3
    width=3
    
    ## define noise region
    nr_dynasquare_sci = region_dynasquare(sci_im_total, sci_signal_i_opp, sci_signal_j_opp, aperture, pix_radius, width, height, opposite=True)
    nr_dynasquare_ref = region_dynasquare(ref_im_total, sci_signal_i_opp, sci_signal_j_opp, aperture, pix_radius, width, height, opposite=True)

    
# =============================================================================
#     ## define noise region
#     nr_dynasquare_sci = region_dynasquare(sci_im_total, sci_plan_i, sci_plan_j, aperture, pix_radius, width, height, opposite=False)
#     nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, ref_im_total, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, aperture, pix_radius, roll_angle)
# =============================================================================


    ## measure noise
    counts_per_ap_nr_dynasquare_sci, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, pix_radius)
    counts_per_ap_nr_dynasquare_ref, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, pix_radius)
    
        
    tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci# - np.median(counts_per_ap_nr_dynasquare_sci)
    tot_ref_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_ref# - np.median(counts_per_ap_nr_dynasquare_ref)    

    


    #signal_ttest, noise_ttest = calc_SNR_ttest(signal_apertures, noise_apertures)

    noise_background = np.mean(tot_sci_ap_counts_dynasquare) + np.mean(tot_ref_ap_counts_dynasquare)
    total_signal_CR = signal_apertures# + np.sum(sci_disk_im*sci_aperture_mask) 
    total_noise_CR = noise_background + total_signal_CR
    total_bkgr_CR = noise_background
    
    tot_tint = target_SNR**2 * total_noise_CR / total_signal_CR**2
    
    
    
    # get noise estimate for outside
    
    sci_signal_i_opp_out, sci_signal_j_opp_out  = get_opp_coords(sci_out_i, sci_out_j, 50)
    
    ## define noise region
    nr_dynasquare_sci_out = region_dynasquare(sci_im_total, sci_signal_i_opp_out, sci_signal_j_opp_out, aperture, pix_radius, width, height, opposite=True)
    nr_dynasquare_ref_out = region_dynasquare(ref_im_total, sci_signal_i_opp_out, sci_signal_j_opp_out, aperture, pix_radius, width, height, opposite=True)

    
# =============================================================================
#     ## define noise region
#     nr_dynasquare_sci = region_dynasquare(sci_im_total, sci_plan_i, sci_plan_j, aperture, pix_radius, width, height, opposite=False)
#     nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, ref_im_total, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, aperture, pix_radius, roll_angle)
# =============================================================================


    ## measure noise
    counts_per_ap_nr_dynasquare_sci_out, ap_coords_nr_dynasquare_sci_out = sum_apertures_in_region(nr_dynasquare_sci_out, aperture, pix_radius)
    counts_per_ap_nr_dynasquare_ref_out, ap_coords_nr_dynasquare_ref_out = sum_apertures_in_region(nr_dynasquare_ref_out, aperture, pix_radius)
    
        
    tot_sci_ap_counts_dynasquare_out = counts_per_ap_nr_dynasquare_sci_out# - np.median(counts_per_ap_nr_dynasquare_sci)
    tot_ref_ap_counts_dynasquare_out = counts_per_ap_nr_dynasquare_ref_out# - np.median(counts_per_ap_nr_dynasquare_ref)    

    


    #signal_ttest, noise_ttest = calc_SNR_ttest(signal_apertures, noise_apertures)

    noise_background_out = np.mean(tot_sci_ap_counts_dynasquare_out) + np.mean(tot_ref_ap_counts_dynasquare_out)
    total_bkgr_CR_out = noise_background_out
    
    
    
    if verbose:
        print("Total Signal Counts:", total_signal_CR*tot_tint)
        print("Total Background Counts:", (total_bkgr_CR )*tot_tint)
        
        print("Total Noise Counts:", (total_noise_CR )*tot_tint)
 
        
        print("tot_tint", tot_tint)
        SNR_calc = total_signal_CR * tot_tint / np.sqrt(total_noise_CR*tot_tint)
        print("SNR_calc:", SNR_calc)
        
        
    
    
    tot_noise_counts = total_noise_CR*tot_tint

    tot_noise_counts_out = total_bkgr_CR_out*tot_tint


    science_image = (sci_plan_im + sci_disk_im + sci_star_im) * tot_tint
    reference_image = ref_star_im * tot_tint
    
    science_image_noisy = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * tot_tint)
    reference_image_noisy = np.random.poisson(ref_star_im * tot_tint)
    
    
    
    if add_noise:
        science_image = science_image_noisy.astype(float)
        reference_image = reference_image_noisy.astype(float)
    
    

    return science_image, reference_image, tot_noise_counts, tot_noise_counts_out, (sci_out_i, sci_out_j)

    
    
    


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
    elif tele == "LUVB":
        shift_order = 0
    
    rad2mas = 180./np.pi*3600.*1000.

    imsc2 = 0.5*0.5e-6/(0.9*diam)*rad2mas # mas
    
    # Compute wavelength-dependent zoom factor.
    fact = 0.25*wave * 1e-6 /diam*rad2mas/imsc2

    norm = np.sum(psf)

    # Scale image to imsc.
    temp = np.exp(zoom(np.log(psf), fact, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
    
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



def rotate_region(region_sci, im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz, roll_angle, opposite=False):
    
    region_ref = ~np.isnan(region_sci)
    region_ref = region_ref.astype(float)
    region_ref[sci_signal_i-ap_sz:sci_signal_i+ap_sz+1, sci_signal_j-ap_sz:sci_signal_j+ap_sz+1] = 1
    
    region_ref = rotate(region_ref, -roll_angle, order=0, reshape=False)
    
    if opposite is False:
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



def calc_SNR_ttest(signal_apertures, noise_apertures):
    
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


    
    signal_ttest = (x1 - x2)
    #signal_ttest = x1
    noise_ttest = (s12*np.sqrt(1/n1 + 1/n2))
    
    return signal_ttest, noise_ttest
    
    
    


def calc_SNR_ttest_ADI(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                       aperture, ap_sz, width, height, roll_angle, corrections=False, verbose=False, out=False):
    
    

    
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
    nr_dynasquare_sci = region_dynasquare(im, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, width, height, opposite=True)
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
    SNR_classic = np.abs((signal_apertures -np.mean(noise_apertures)))  / np.sqrt(np.std(noise_apertures, ddof=1)**2 + (signal_apertures -np.mean(noise_apertures)))
    #noise_apertures = np.concatenate((tot_sci_ap_counts_dynasquare, -1*tot_ref_ap_counts_dynasquare))
    std_correction_factor = np.sqrt(1 + (1/len(noise_apertures)))
    measured_noise = np.nanstd(noise_apertures, ddof=1) * std_correction_factor
    
    #print((signal_apertures - np.mean(noise_apertures)) / np.sqrt(measured_noise**2 + signal_apertures))
    
    signal_ttest, noise_ttest = calc_SNR_ttest(signal_apertures, noise_apertures)
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
    
    return SNR_total, SNR_classic, signal_ttest, total_noise, noise_map_sci

def calc_SNR_ttest_RDI(im, im_hipass, sci_signal_i, sci_signal_j, sci_signal_i_opp, sci_signal_j_opp, 
                       aperture, ap_sz, width, height, roll_angle, corrections=True, verbose=False):
    
    imsz, imsz = im_hipass.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    sci_signal_mask = np.zeros_like(im_hipass, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    
    sci_sig = im[sci_signal_mask]
    
    signal_apertures = np.sum(sci_sig) 
    
# =============================================================================
#     print(signal_apertures)
#     assert False
# =============================================================================
    
    
    ## define noise region
    nr_dynasquare_sci = region_dynasquare(im_hipass, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, width, height, opposite=True)
    
    if corrections:
        #### do an r^2 correction on the region
        nr_dynasquare_sci = r2_correction(nr_dynasquare_sci)
    
    ## measure noise
    counts_per_ap_nr_dynasquare_sci, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    
        
    tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci
    
    
    noise_apertures = tot_sci_ap_counts_dynasquare 
    
    std_correction_factor = np.sqrt(1 + (1/len(noise_apertures)))
    measured_noise = np.nanstd(noise_apertures, ddof=1) * std_correction_factor
    
    SNR_classic = np.abs((signal_apertures -np.mean(noise_apertures)))  / np.sqrt(np.std(noise_apertures, ddof=1)**2 + (signal_apertures -np.mean(noise_apertures)))

    
    signal_ttest, noise_ttest = calc_SNR_ttest(signal_apertures, noise_apertures)
    
    total_noise = np.sqrt(noise_ttest**2 + signal_apertures)

    
    SNR_total = signal_ttest / total_noise

    
    noise_map_sci = ~np.isnan(nr_dynasquare_sci) 
    
    return SNR_total, SNR_classic, signal_ttest, total_noise, noise_map_sci


