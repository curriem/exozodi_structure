import numpy as np
#import coronagraph
#import detector
#import util
import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, correlate2d
from scipy.ndimage import rotate, zoom, shift
import glob
import os
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import interp1d

def plot_im(im, signal_i, signal_j, log=False):
    plt.figure(figsize=(20,20))
    if log:
        plt.imshow(np.log(im), origin='lower')
    else:
        plt.imshow(im, origin='lower')
    plt.axhline(signal_i, color="white", linestyle="--")
    plt.axvline(signal_j, color="white", linestyle="--")
    plt.colorbar()
def plot_im_ADI(im, im1_signal_i, im1_signal_j, im2_signal_i, im2_signal_j):
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

    Args:
        img: a 2D image
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Returns:
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


def cc_SNR_unknown_loc(cc_map, pix_radius, mask_antisignal=False):
    signal_i, signal_j = np.unravel_index(np.nanargmax(cc_map), cc_map.shape)
    signal_mask = np.zeros_like(cc_map, dtype=bool)
    
    signal_size = 2*pix_radius
    signal_mask[signal_i-signal_size:signal_i+signal_size+1, signal_j-signal_size:signal_j+signal_size+1] = True
    
    if mask_antisignal:
        valid_cc_mask = calculate_valid_cc_mask(cc_map, signal_i, signal_j, roll_angle, aperture, central_pixel)
        cc_sig = (cc_map[signal_i, signal_j])# - np.nanmean(cc_map[valid_cc_mask])) 
#         cc_bkgr = np.nanstd(cc_map_single[valid_cc_mask] )
        cc_bkgr = np.nanstd(cc_map[valid_cc_mask] )
    else:
        cc_sig = (cc_map[signal_i, signal_j])# - np.nanmean(cc_map[~signal_mask]))  
#      cc_bkgr = np.nanstd(cc_map_single[~signal_mask] )
        cc_bkgr = np.nanstd(cc_map[~signal_mask] )
    
    #print("cc_sig", cc_sig)
    cc_SNR = cc_sig / cc_bkgr
#     cc_SNR = cc_sig / np.sqrt(cc_bkgr)
    return cc_SNR



def calculate_cc_map(matched_filter_datacube, im, valid_mask, hipass=False, filtersize=None):
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
    # sep_mas: planet--star separation in mas
    # lam: wl of observation in um
    # D: diam of telescope in m
    
    # returns lamD: separation in lam/D
    lam = lam.to(u.m)
    sep_lamD = (D/lam) * (1/u.radian) * sep_mas.to(u.radian)
    return sep_lamD

def lamD_to_mas(lamD_sep, lam, D):
    # sep_lamD: planet--star separation in lamD
    # lam: wl of observation in um
    # D: diam of telescope in m
    # returns sep_mas: separation in mas
    
    lam = lam.to(u.m)
    sep_mas = (lamD_sep * (lam / D) * u.radian).to(u.mas)
    
    return sep_mas

def construct_maps(arr, pixscale_mas, diam, IWA_lamD=8.5, OWA_lamD=26., plotting=False):
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
    
    x_min = int(psf_i-pix_radius)
    x_max = int(psf_i+pix_radius)+1
    
    y_min = int(psf_j-pix_radius)
    y_max = int(psf_j+pix_radius) +1
    #print(x_min, x_max)
    psf_stamp = psf[x_min:x_max, y_min:y_max]
    #print(psf_stamp)
    stamp_center = pix_radius
    
    for i in range(pix_radius*2 + 1):
        for j in range(pix_radius*2 + 1):
            
            dist_from_center = np.sqrt((i-stamp_center)**2 + (j-stamp_center)**2)
            if dist_from_center > pix_radius:
                psf_stamp[i, j] = 0
    
    return psf_stamp



def synthesize_images(im_dir, sci_plan_i, sci_plan_j, ref_plan_i, ref_plan_j, zodis, aperture, target_SNR=7, pix_radius=1, verbose=False, planet_noise=True, add_noise=True, add_star=True, uniform_disk=False, r2_disk=False):
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
    
#     plot_im(aperture_mask, plan_i, plan_j)
    
    sci_plan_CR = np.sum(sci_plan_im * sci_aperture_mask)
    ref_plan_CR = np.sum(ref_plan_im * ref_aperture_mask)
    
    if add_star:
        sci_star_CR = np.sum(sci_star_im * sci_aperture_mask)
        ref_star_CR = np.sum(ref_star_im * ref_aperture_mask)
    else:
        sci_star_CR = 0
        sci_star_im = np.zeros_like(sci_star_im)
        ref_star_CR = 0
        ref_star_im = np.zeros_like(ref_star_im)
        
    sci_disk_im_fits = pyfits.open(im_dir + "/DET/sci_disk.fits")
    sci_disk_im = sci_disk_im_fits[0].data[0, 0]
    
    ref_disk_im_fits = pyfits.open(im_dir + "/DET/ref_disk.fits")
    ref_disk_im = ref_disk_im_fits[0].data[0, 0]
    

    if uniform_disk:
        median_disk_val = 0.1#np.max(sci_disk_im)
        sci_disk_im = median_disk_val * np.ones_like(sci_disk_im)
        ref_disk_im = median_disk_val * np.ones_like(ref_disk_im)
        
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
        
    
    sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
    sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
    
    ref_disk_CR = np.sum(ref_disk_im*ref_aperture_mask)
    ref_disk_CR = np.sum(ref_disk_im*ref_aperture_mask)
    
    sci_back_CR = sci_disk_CR + sci_star_CR # ph/s
    ref_back_CR = ref_disk_CR + ref_star_CR # ph/s

    
    # tot_noise_CR = 2.*sci_back_CR # ph/s
    tot_noise_CR = sci_back_CR + ref_back_CR # ph/s
# =============================================================================
#     sci_noise_CR = sci_back_CR
#     ref_noise_CR = ref_back_CR
# =============================================================================
#     noise_CR = 1.*back_CR # ph/s
    if planet_noise:
        tot_noise_CR += sci_plan_CR
# =============================================================================
#         sci_noise_CR += sci_plan_CR
#         ref_noise_CR += ref_plan_CR
# =============================================================================
        
    tot_tint = target_SNR**2 * tot_noise_CR/sci_plan_CR**2 # s
    
# =============================================================================
#     SNR_per_image = target_SNR / np.sqrt(2)
# =============================================================================
    
# =============================================================================
#     sci_tint = SNR_per_image**2 * sci_noise_CR/sci_plan_CR**2 # s
#     ref_tint = SNR_per_image**2 * ref_noise_CR/ref_plan_CR**2 # s
# =============================================================================

    sci_tint = tot_tint/2
    ref_tint = tot_tint/2
    #print(tot_tint, sci_tint, ref_tint)
    #assert False
    
#     tint /= 2
    
    if verbose:
        print("Sci planet counts:", sci_plan_CR*sci_tint)
        print("Sci Disk counts:", sci_disk_CR*sci_tint)
        print("Sci Star counts:", sci_star_CR*sci_tint)
        print("Sci Integration time:", sci_tint)
        
        print("Ref planet counts:", ref_plan_CR*ref_tint)
        print("Ref Disk counts:", ref_disk_CR*ref_tint)
        print("Ref Star counts:", ref_star_CR*ref_tint)
        print("Ref Integration time:", ref_tint)
        
    sci_planet_counts, ref_planet_counts = sci_plan_CR*sci_tint, ref_plan_CR*ref_tint

    science_image = (sci_plan_im + sci_disk_im + sci_star_im) * sci_tint
    reference_image = (ref_plan_im + ref_disk_im + ref_star_im) * ref_tint
    
    if ~planet_noise:
        science_poisson = np.random.poisson((sci_disk_im + sci_star_im) * sci_tint)
        reference_poisson = np.random.poisson((ref_disk_im + ref_star_im) * ref_tint)
    elif planet_noise:
        science_poisson = np.random.poisson((sci_disk_im + sci_star_im + sci_plan_im) * sci_tint)
        reference_poisson = np.random.poisson((ref_disk_im + ref_star_im + ref_plan_im) * ref_tint)
#     plot_im(sci_disk_im * tint + np.random.poisson(sci_disk_im*tint) + sci_plan_im*tint, 50, 50)
#     plt.title("This one")
    
    

    if add_noise:
        science_image += science_poisson
        reference_image += reference_poisson
    
#     plot_im(science_image, 50, 50)
#     plt.title("This one is the bad one")
    return science_image, reference_image, sci_planet_counts, ref_planet_counts


def calculate_SNR(im, signal_i, signal_j, valid_map, aperture, region_radius, r2_correct=False, force_signal=None):
    
    imsz, imsz = im.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    signal_mask = np.zeros_like(im, dtype=bool)
    signal_mask[signal_i-ap_rad:signal_i+ap_rad+1, signal_j-ap_rad: signal_j+ap_rad+1] = aperture
    
    
    valid_map_signal = np.ones_like(im, dtype=bool)
    valid_map_signal[signal_i-2*ap_rad:signal_i+2*ap_rad+1, signal_j-2*ap_rad: signal_j+2*ap_rad+1] = False
    
    valid_map = valid_map & valid_map_signal

    
    # get noise region

    
    noise_region = calculate_noise_region_adjacent(im, signal_i, signal_j, aperture, ap_rad, region_radius, offset=10)
    
    if r2_correct:
        noise_region = r2_correction(noise_region, signal_i, signal_j)
    noise_region_median = np.nanmedian(noise_region)
    
    noise_region_bkgr_rm = noise_region - noise_region_median
    
    noise = calc_noise_in_region(noise_region_bkgr_rm, aperture, ap_rad)
    
    im_bkgr_sub = im - noise_region_median
    
    signal = np.sum(im_bkgr_sub[signal_mask])
    
    SNR = signal / noise
    
    if force_signal is not None:
        SNR = force_signal / noise
    
    return SNR

def calculate_SNR_sci_ref(sci_im, ref_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, valid_map, aperture, region_radius, force_signal=None):
    
    imsz, imsz = sci_im.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    sci_signal_mask = np.zeros_like(sci_im, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    ref_signal_mask = np.zeros_like(ref_im, dtype=bool)
    ref_signal_mask[ref_signal_i-ap_rad:ref_signal_i+ap_rad+1, ref_signal_j-ap_rad: ref_signal_j+ap_rad+1] = aperture
    
    
    valid_map_signal = np.ones_like(sci_im, dtype=bool)
    valid_map_signal[sci_signal_i-2*ap_rad:sci_signal_i+2*ap_rad+1, sci_signal_j-2*ap_rad: sci_signal_j+2*ap_rad+1] = False
    valid_map_signal[ref_signal_i-2*ap_rad:ref_signal_i+2*ap_rad+1, ref_signal_j-2*ap_rad: ref_signal_j+2*ap_rad+1] = False
    
    valid_map = valid_map & valid_map_signal

    
    # get noise region

    
    sci_noise_region = calculate_noise_region_adjacent(sci_im, sci_signal_i, sci_signal_j, aperture, ap_rad, region_radius, offset=10)
    ref_noise_region = calculate_noise_region_adjacent(ref_im, sci_signal_i, sci_signal_j, aperture, ap_rad, region_radius, offset=10)
    
    sci_noise_region_median = np.nanmedian(sci_noise_region)
    sci_noise_region_bkgr_rm = sci_noise_region - sci_noise_region_median
    
    ref_noise_region_median = np.nanmedian(ref_noise_region)
    ref_noise_region_bkgr_rm = ref_noise_region - ref_noise_region_median
    
    noise = calc_noise_in_region(sci_noise_region_bkgr_rm + ref_noise_region_bkgr_rm, aperture, ap_rad)
    
    sci_im_bkgr_sub = sci_im - sci_noise_region_median
    ref_im_bkgr_sub = ref_im - ref_noise_region_median
#     plot_im(im_bkgr_sub, signal_i, signal_j)
    sci_signal = np.sum(sci_im_bkgr_sub[sci_signal_mask])
    ref_signal = np.sum(ref_im_bkgr_sub[ref_signal_mask])
    
    signal = sci_signal+ref_signal

#     print("noise", noise)
#     print("signal", signal)
#     print("SNR",signal/noise)
    
#     plot_im(noise_region_bkgr_rm, 50, 50)
#     plt.title("Noise region")

    
    SNR = signal / noise
    
    if force_signal is not None:
        SNR = force_signal / noise
    
    return SNR

def calculate_SNR_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, valid_map, aperture, region_radius, r2_correct=True, force_signal=None):
    
    imsz, imsz = sub_im.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    sci_signal_mask = np.zeros_like(sub_im, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    ref_signal_mask = np.zeros_like(sub_im, dtype=bool)
    ref_signal_mask[ref_signal_i-ap_rad:ref_signal_i+ap_rad+1, ref_signal_j-ap_rad: ref_signal_j+ap_rad+1] = aperture

    
    # get noise region

    
    sub_noise_region = calculate_noise_region_adjacent(sub_im, sci_signal_i, sci_signal_j, aperture, ap_rad, region_radius, offset=10)
    
    sub_noise_region_median = np.nanmedian(sub_noise_region)
    sub_noise_region_bkgr_rm = sub_noise_region - sub_noise_region_median
    
    if r2_correct:
        sub_noise_region_bkgr_rm = r2_correction(sub_noise_region_bkgr_rm, sci_signal_i, sci_signal_j)
    
    # noise = calc_noise_in_region(sub_noise_region_bkgr_rm, aperture, ap_rad)
    noise_two_aps = calc_noise_in_region_two_apertures(sub_noise_region_bkgr_rm, aperture, ap_rad)
    noise = noise_two_aps
    
    sub_im_bkgr_sub = sub_im - sub_noise_region_median
#     plot_im(im_bkgr_sub, signal_i, signal_j)
    sci_signal = np.sum(sub_im_bkgr_sub[sci_signal_mask])
    ref_signal = np.sum( -1 * sub_im_bkgr_sub[ref_signal_mask])
    
    signal = sci_signal+ref_signal

#     print("noise", noise)
#     print("signal", signal)
#     print("SNR",signal/noise)
    
#     plot_im(noise_region_bkgr_rm, 50, 50)
#     plt.title("Noise region")

    
    SNR = signal / noise
    
    if force_signal is not None:
        SNR = force_signal / noise
    
    return SNR

def calc_noise_in_region(im, aperture, ap_rad):
    # im should have non-noise regions masked off
    imsz, imsz = im.shape
    
    background_vals = []
    
    non_nan_inds = np.where(~np.isnan(im))
    i_arr, j_arr = non_nan_inds
    for n in range(len(i_arr)):
        i = i_arr[n]
        j = j_arr[n]
        background_aperture = im[i-ap_rad:i+ap_rad+1, j-ap_rad:j+ap_rad+1] * aperture
        background_aperture_sum = np.sum(background_aperture)
        background_vals.append(background_aperture_sum)
    
    region_std = np.nanstd(background_vals)
    
    return region_std

def calc_noise_in_region_two_apertures(im, aperture, ap_rad):
    # im should have non-noise regions masked off
    imsz, imsz = im.shape
    
    background_vals = []
    non_nan_inds = np.where(~np.isnan(im))
    i_arr, j_arr = non_nan_inds

    for n in range(len(i_arr)):
        for m in range(len(i_arr)):
            i1 = i_arr[n]
            j1 = j_arr[n]
            i2 = i_arr[m]
            j2 = j_arr[m]
            
            if (i1 == i2) and (j1 == j2):
                background_vals.append(np.nan)
            else:
                background_aperture1 = im[i1-ap_rad:i1+ap_rad+1, j1-ap_rad:j1+ap_rad+1] * aperture
                background_aperture2 = -1 * im[i2-ap_rad:i2+ap_rad+1, j2-ap_rad:j2+ap_rad+1] * aperture
                background_aperture_sum = np.sum(background_aperture1 + background_aperture2)
                background_vals.append(background_aperture_sum)
    
    region_std = np.nanstd(background_vals)
    
    return region_std

def calculate_noise_region_adjacent(im, signal_i, signal_j, aperture, ap_rad, region_radius, offset=10):
    imsz, imsz = im.shape
    imctr = (imsz-1)/2
    
    adjacent_mask = np.ones_like(im) * np.nan
    
    signal_angle = np.arctan2((signal_i-imctr), (signal_j-imctr))
    
    d_plan = np.sqrt((signal_i-imctr)**2 + (signal_j-imctr)**2)
    
    region_dist = d_plan + offset
    
    noise_i = region_dist * np.sin(signal_angle) + imctr
    noise_j = region_dist * np.cos(signal_angle) + imctr
    
    for i in range(imsz):
        for j in range(imsz):
            d = np.sqrt((i-noise_i)**2 + (j-noise_j)**2)
            if d < region_radius:
                adjacent_mask[i,j] = 1. 
        
    # mask out signal
    adjacent_mask[signal_i-ap_rad:signal_i+ap_rad+1, signal_j-ap_rad:signal_j+ap_rad+1] = np.nan
    
    noise_region = im*adjacent_mask
    
    return noise_region


def r2_correction(im, signal_i, signal_j):
    imsz, imsz = im.shape
    imctr = (imsz-1)/2
    
    im_corrected = np.empty_like(im)
    
    signal_dist = np.sqrt((signal_i-imctr)**2 + (signal_j-imctr)**2)
    
    for i in range(imsz):
        for j in range(imsz):
            dist = np.sqrt((i-imctr)**2 + (j-imctr)**2)
            
            correction = dist / signal_dist
            
            im_corrected[i,j] = correction * im[i,j]
    return im_corrected

def cc_SNR_known_loc(cc_map, signal_i, signal_j, pix_radius, roll_angle, aperture, central_pixel, region_radius, r2_correct=True, mask_antisignal=True):
    signal_mask = np.zeros_like(cc_map, dtype=bool)
    signal_size = 2*pix_radius
    signal_mask[signal_i-signal_size:signal_i+signal_size+1, signal_j-signal_size:signal_j+signal_size+1] = True
    
    if mask_antisignal:
        valid_cc_mask = calculate_valid_cc_mask(cc_map, signal_i, signal_j, roll_angle, aperture, central_pixel)
        cc_noise_region = calculate_noise_region_adjacent(cc_map, signal_i, signal_j, aperture, pix_radius, region_radius, offset=10)
        
        if r2_correct:
            cc_noise_region = r2_correction(cc_noise_region, signal_i, signal_j)
        
        cc_sig = cc_map[signal_i, signal_j]# - np.nanmean(cc_map[valid_cc_mask]))
        
        # use this one for calculating std of entire cc map minus sig/antisig
        #cc_bkgr = np.nanstd(cc_map[valid_cc_mask] )
        
        # use this one for calculating std of region adjacent to known sig
        cc_bkgr = np.nanstd(cc_noise_region)
    else:
        cc_sig = (cc_map[signal_i, signal_j])# - np.nanmean(cc_map[~signal_mask]))
#         cc_bkgr = np.nanstd(cc_map_single[~signal_mask])
        cc_bkgr = np.nanstd(cc_map[~signal_mask])
    
    cc_SNR = cc_sig / cc_bkgr
#     print("cc_sig", cc_sig)
#     print("cc_bkgr", cc_bkgr)


#     cc_SNR = cc_sig / np.sqrt(cc_bkgr)
    return cc_SNR

def calculate_valid_cc_mask(cc_map, signal_i, signal_j, roll_angle, aperture, central_pixel):
    apsz, apsz = aperture.shape
    signal_size = 4*(apsz - 1)
    nan_mask = ~np.isnan(cc_map)
    
    # signal location
    signal_mask = np.ones_like(cc_map, dtype=bool)
    signal_mask[signal_i-signal_size:signal_i+signal_size+1, signal_j-signal_size:signal_j+signal_size+1] = False
        
    dist_from_center_pix = np.sqrt((signal_i-central_pixel)**2 + (signal_j-central_pixel)**2)

    # antisignal 1 location
    antisignal1_x2 = dist_from_center_pix * np.sin(np.deg2rad(roll_angle))
    antisignal1_y2 = dist_from_center_pix * np.cos(np.deg2rad(roll_angle))
    antisignal1_i = round(antisignal1_x2 + central_pixel)
    antisignal1_j = round(antisignal1_y2 + central_pixel)
    
    antisignal1_mask = np.ones_like(cc_map, dtype=bool)
    antisignal1_mask[antisignal1_i-signal_size:antisignal1_i+signal_size+1, antisignal1_j-signal_size:antisignal1_j+signal_size+1] = False

    
    # antisignal 2 location
    antisignal2_x2 = dist_from_center_pix * np.sin(np.deg2rad(-roll_angle))
    antisignal2_y2 = dist_from_center_pix * np.cos(np.deg2rad(-roll_angle))
    antisignal2_i = round(antisignal2_x2 + central_pixel)
    antisignal2_j = round(antisignal2_y2 + central_pixel)
    
    antisignal2_mask = np.ones_like(cc_map, dtype=bool)
    antisignal2_mask[antisignal2_i-signal_size:antisignal2_i+signal_size+1, antisignal2_j-signal_size:antisignal2_j+signal_size+1] = False

    valid_mask = nan_mask & signal_mask & antisignal1_mask & antisignal2_mask
    
    

    return valid_mask


def downbin_psf(psf, imsc, imsz, wave, diam, tele):
    
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



