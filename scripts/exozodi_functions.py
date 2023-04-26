import numpy as np
import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, shift, rotate
from scipy.interpolate import NearestNDInterpolator

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


# =============================================================================
# def cc_SNR_unknown_loc(cc_map, pix_radius, mask_antisignal=False):
#     signal_i, signal_j = np.unravel_index(np.nanargmax(cc_map), cc_map.shape)
#     signal_mask = np.zeros_like(cc_map, dtype=bool)
#     
#     signal_size = 2*pix_radius
#     signal_mask[signal_i-signal_size:signal_i+signal_size+1, signal_j-signal_size:signal_j+signal_size+1] = True
#     
#     if mask_antisignal:
#         valid_cc_mask = calculate_valid_cc_mask(cc_map, signal_i, signal_j, roll_angle, aperture, central_pixel)
#         cc_sig = (cc_map[signal_i, signal_j])# - np.nanmean(cc_map[valid_cc_mask])) 
# #         cc_bkgr = np.nanstd(cc_map_single[valid_cc_mask] )
#         cc_bkgr = np.nanstd(cc_map[valid_cc_mask] )
#     else:
#         cc_sig = (cc_map[signal_i, signal_j])# - np.nanmean(cc_map[~signal_mask]))  
# #      cc_bkgr = np.nanstd(cc_map_single[~signal_mask] )
#         cc_bkgr = np.nanstd(cc_map[~signal_mask] )
#     
#     #print("cc_sig", cc_sig)
#     cc_SNR = cc_sig / cc_bkgr
# #     cc_SNR = cc_sig / np.sqrt(cc_bkgr)
#     return cc_SNR
# =============================================================================



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
                          add_noise=True, add_star=True, uniform_disk=False, r2_disk=False, simple_planet=False,
                          sci_noise_coords=None, ref_noise_coords=None):
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
        
    
        
    if sci_noise_coords is not None:
        
        # calculating the average CRs in a region around the planet
        # this is how SNR is measured later, so it makes sense to use now
        

# =============================================================================
#         sci_star_CRs = []
#         sci_disk_CRs = []
#         for sci_coords in sci_noise_coords:
#             coord_i, coord_j = sci_coords
#             
#             sci_star_aperture = sci_star_im[coord_i-pix_radius:coord_i+pix_radius+1, coord_j-pix_radius:coord_j+pix_radius+1]
#             sci_star_aperture *= aperture
#             sci_star_CRs.append(np.sum(sci_star_aperture))
#             
#             sci_disk_aperture = np.copy(sci_disk_im[coord_i-pix_radius:coord_i+pix_radius+1, coord_j-pix_radius:coord_j+pix_radius+1])
#             sci_disk_aperture *= aperture
#             
#             sci_disk_CRs.append(np.sum(sci_disk_aperture))
#             
#         
#         sci_star_CR = np.median(sci_star_CRs)
#         sci_disk_CR = np.median(sci_disk_CRs)
#         
#         ref_star_CRs = []
#         ref_disk_CRs = []
#         for ref_coords in ref_noise_coords:
#             coord_i, coord_j = ref_coords
#             
#             ref_star_aperture = ref_star_im[coord_i-pix_radius:coord_i+pix_radius+1, coord_j-pix_radius:coord_j+pix_radius+1]
#             ref_star_aperture *= aperture
#             ref_star_CRs.append(np.sum(ref_star_aperture))
#             
#             ref_disk_aperture = ref_disk_im[coord_i-pix_radius:coord_i+pix_radius+1, coord_j-pix_radius:coord_j+pix_radius+1]
#             ref_disk_aperture *= aperture
#             
#             ref_disk_CRs.append(np.sum(ref_disk_aperture))
#             
#         
#         ref_star_CR = np.median(ref_star_CRs)
#         ref_disk_CR = np.median(ref_disk_CRs)
# =============================================================================

        CR_rad = 4
        sci_star_CR = 5*np.median(sci_star_im[sci_plan_i-CR_rad:sci_plan_i+CR_rad+1, sci_plan_j-CR_rad:sci_plan_j+CR_rad+1])
        sci_disk_CR = 5* np.median(sci_disk_im[sci_plan_i-CR_rad:sci_plan_i+CR_rad+1, sci_plan_j-CR_rad:sci_plan_j+CR_rad+1])
        
        ref_star_CR = 5*np.median(ref_star_im[ref_plan_i-CR_rad:ref_plan_i+CR_rad+1, ref_plan_j-CR_rad:ref_plan_j+CR_rad+1])
        ref_disk_CR = 5*np.median(ref_disk_im[ref_plan_i-CR_rad:ref_plan_i+CR_rad+1, ref_plan_j-CR_rad:ref_plan_j+CR_rad+1])

        sci_plan_noise_region = np.copy(sci_plan_im[sci_plan_i-CR_rad:sci_plan_i+CR_rad+1, sci_plan_j-CR_rad:sci_plan_j+CR_rad+1])
        print(sci_plan_noise_region)
        sci_plan_noise_region[sci_plan_i-pix_radius:sci_plan_i+pix_radius+1, sci_plan_j-CR_rad:sci_plan_j+CR_rad+1] *= ~aperture
        print(sci_plan_noise_region)
        assert False

        tot_bkgr_CR = sci_star_CR + sci_disk_CR + ref_star_CR + ref_disk_CR
        
        total_signal_CR = sci_plan_CR + ref_plan_CR
        total_noise_CR = total_signal_CR + tot_bkgr_CR


        sci_star_CR_out = np.sum(sci_star_im * sci_aperture_mask_out)
        ref_star_CR_out = np.sum(ref_star_im * ref_aperture_mask_out)
        sci_disk_CR_out = np.sum(sci_disk_im*sci_aperture_mask_out)
        ref_disk_CR_out = np.sum(ref_disk_im*ref_aperture_mask_out)
        sci_bkgr_CR_out = sci_disk_CR_out + sci_star_CR_out
        ref_bkgr_CR_out = ref_disk_CR_out + ref_star_CR_out
        tot_bkgr_CR_out = sci_bkgr_CR_out + ref_bkgr_CR_out
            
            
            
     
            
    
    else:
    
    
        sci_star_CR = np.sum(sci_star_im * sci_aperture_mask)
        ref_star_CR = np.sum(ref_star_im * ref_aperture_mask)
        
        sci_star_CR_out = np.sum(sci_star_im * sci_aperture_mask_out)
        ref_star_CR_out = np.sum(ref_star_im * ref_aperture_mask_out)
        
        
        
        sci_disk_CR = np.sum(sci_disk_im*sci_aperture_mask)
        ref_disk_CR = np.sum(ref_disk_im*ref_aperture_mask)

        sci_disk_CR_out = np.sum(sci_disk_im*sci_aperture_mask_out)
        ref_disk_CR_out = np.sum(ref_disk_im*ref_aperture_mask_out)
        
# =============================================================================
#         print("sci star CR:", sci_star_CR)
#         print("ref star CR:", ref_star_CR)
#         print("sci disk CR:", sci_disk_CR)
#         print("ref disk CR:", ref_disk_CR)
#         assert False
# =============================================================================

    
    
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
    
    
# =============================================================================
#     plt.figure()
#     plt.imshow(science_image[sci_plan_i-7:sci_plan_i+8, sci_plan_j-7:sci_plan_j+8] - reference_image[sci_plan_i-7:sci_plan_i+8, sci_plan_j-7:sci_plan_j+8])
#     plt.colorbar()
#     assert False
# =============================================================================
    

    

    return science_image, reference_image, sci_planet_counts, ref_planet_counts, tot_noise_counts, tot_noise_counts_out, (sci_out_i, sci_out_j, ref_out_i, ref_out_j)


def synthesize_images_RDI(im_dir, sci_plan_i, sci_plan_j, zodis, aperture,
                          target_SNR=7, pix_radius=1, verbose=False, planet_noise=True, 
                          add_noise=True, add_star=True, uniform_disk=False, r2_disk=False):
    
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

def region_ring(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz):
    
    imsz, imsz = sub_im.shape
    imctr = (imsz-1)/2
    
    dr = 2*(2*ap_sz+1)
    
    noise_mask = np.zeros_like(sub_im)  
    
    signal_rad = np.sqrt((sci_signal_i-imctr)**2 + (sci_signal_j-imctr)**2)
    inner_r = signal_rad - dr/2
    outer_r = signal_rad + dr/2
    
    for i in range(imsz):
        for j in range(imsz):
            dist = np.sqrt((i-imctr)**2 + (j-imctr)**2)
            if dist >= inner_r and dist < outer_r:
                noise_mask[i, j] = 1
                
    noise_mask[sci_signal_i-ap_sz:sci_signal_i+ap_sz+1, sci_signal_j-ap_sz:sci_signal_j+ap_sz+1] = ~aperture
    noise_mask[ref_signal_i-ap_sz:ref_signal_i+ap_sz+1, ref_signal_j-ap_sz:ref_signal_j+ap_sz+1] = ~aperture
    
    zero_inds = np.where(noise_mask == 0.)
    
    noise_mask[zero_inds] = np.nan

    noise_region = sub_im * noise_mask

    return noise_region

def region_circle(im, signal_i, signal_j, aperture, ap_sz, opposite=False):
    
    imsz, imsz = im.shape
    imctr = (imsz-1)/2
    
    if opposite:
        inner_r = -1
    else:
        inner_r = ap_sz

    outer_r = 2*(2*ap_sz+1)
    
    noise_mask = np.ones_like(im) * np.nan
    
    for i in range(imsz):
        for j in range(imsz):
            d = np.sqrt((i-signal_i)**2 + (j-signal_j)**2)
            if (d > inner_r) and (d < outer_r):
                noise_mask[i, j] = 1.
                
    noise_region = im*noise_mask
    
    return noise_region

def region_wedge(sub_im, signal_i, signal_j, aperture, ap_sz, rotation_map, angle, tele, disk_region, region_len_r=None, opposite=False):
    
    imsz, imsz = sub_im.shape
    imctr = (imsz-1)/2
    if region_len_r is None:
        if tele == "LUVA":
            if disk_region == "struct":
                lower_dist = 17
                upper_dist = 25
            elif disk_region == "smooth":
                lower_dist = 36
                upper_dist = 44
        elif tele == "LUVB":
            if disk_region == "struct":
                lower_dist = 10
                upper_dist = 18
            elif disk_region == "smooth":
                lower_dist = 26
                upper_dist = 34
    else:
        sig_dist_frm_ctr = np.sqrt((signal_i - imctr)**2 + (signal_j - imctr)**2)
        lower_dist = sig_dist_frm_ctr - region_len_r
        upper_dist = sig_dist_frm_ctr + region_len_r
        #print("lower, upper:", lower_dist, upper_dist)
        

    
    noise_mask = np.zeros_like(sub_im)
    
    sig_rot = rotation_map[signal_i, signal_j]
    
    lower_ang = sig_rot-angle/2
    upper_ang = sig_rot+angle/2
    
    
# =============================================================================
#     sig_dist = np.sqrt((signal_i-imctr)**2 + (signal_j-imctr)**2) 
#     
#     lower_dist = sig_dist - (ap_sz*2+1)
#     upper_dist = sig_dist + (ap_sz*2+1)
# =============================================================================
    
    
    for i in range(imsz):
        for j in range(imsz):
            ang = rotation_map[i,j]
            dist = np.sqrt((i-imctr)**2 + (j-imctr)**2)
            if ang >= lower_ang and ang <= upper_ang and dist >= lower_dist and dist <= upper_dist:
                noise_mask[i, j] = 1
    
    if opposite:
        pass
    else:
        noise_mask[signal_i-ap_sz:signal_i+ap_sz+1, signal_j-ap_sz:signal_j+ap_sz+1] = ~aperture


    zero_inds = np.where(noise_mask == 0.)
    noise_mask[zero_inds] = np.nan
    
    noise_region = sub_im * noise_mask
    
    return noise_region

def region_dynasquare(sub_im, signal_i, signal_j, aperture, ap_sz, width, height, opposite=False):
    
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


def get_opposite_wedge_region(wedge_region, im, signal_i_opp, signal_j_opp, ap_sz):
    
    imsz, imsz = wedge_region.shape
    imctr = (imsz-1)/2

    region_inds = ~np.isnan(wedge_region)
    opp_region = np.zeros_like(wedge_region)
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



def measure_noise_circle_ADI(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                             sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                             aperture, ap_sz):
    ## define noise region
    nr_circle_sci = region_circle(im, sci_signal_i, sci_signal_j, aperture, ap_sz)
    nr_circle_sci_opp = region_circle(im, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, opposite=True)
    nr_circle_ref = region_circle(im, ref_signal_i, ref_signal_j, aperture, ap_sz)
    nr_circle_ref_opp = region_circle(im, ref_signal_i_opp, ref_signal_j_opp, aperture, ap_sz, opposite=True)
    
    ## measure noise
    counts_per_ap_nr_circle_sci, ap_coords_nr_circle_sci = sum_apertures_in_region(nr_circle_sci, aperture, ap_sz)
    counts_per_ap_nr_circle_sci_opp, ap_coords_nr_circle_sci_opp = sum_apertures_in_region(nr_circle_sci_opp, aperture, ap_sz)
    counts_per_ap_nr_circle_ref, ap_coords_nr_circle_ref = sum_apertures_in_region(nr_circle_ref, aperture, ap_sz)
    counts_per_ap_nr_circle_ref_opp, ap_coords_nr_circle_ref_opp = sum_apertures_in_region(nr_circle_ref_opp, aperture, ap_sz)    

    #### total noise counts
    tot_sci_ap_counts_circle = np.concatenate((counts_per_ap_nr_circle_sci, counts_per_ap_nr_circle_sci_opp))
    tot_ref_ap_counts_circle = np.concatenate((counts_per_ap_nr_circle_ref, counts_per_ap_nr_circle_ref_opp))
    
    ##### check if sci and ref regions have equal number of apertures sampled
    if len(tot_sci_ap_counts_circle) == len(tot_ref_ap_counts_circle):
        pass
    elif len(tot_sci_ap_counts_circle) > len(tot_ref_ap_counts_circle):
        num_inds_to_cut = len(tot_sci_ap_counts_circle) - len(tot_ref_ap_counts_circle)
        tot_sci_ap_counts_circle = tot_sci_ap_counts_circle[:-num_inds_to_cut]
    elif len(tot_sci_ap_counts_circle) < len(tot_ref_ap_counts_circle):
        num_inds_to_cut = len(tot_ref_ap_counts_circle) - len(tot_sci_ap_counts_circle)
        tot_ref_ap_counts_circle = tot_ref_ap_counts_circle[:-num_inds_to_cut]
    else:
        assert False, "Something is wrong"
        
    tot_noise_counts_circle = tot_sci_ap_counts_circle + -1 * tot_ref_ap_counts_circle

    #### sigma clip 
    tot_noise_counts_circle_sgcl = sigma_clip(tot_noise_counts_circle)
    
    measured_noise_circle = np.nanstd(tot_noise_counts_circle_sgcl, ddof=1)
    
    #print("Apertures sampled:", len(tot_noise_counts_circle))
    #print("Measured noise circle", measured_noise_circle)
    
    return measured_noise_circle, nr_circle_sci, nr_circle_sci_opp, nr_circle_ref, nr_circle_ref_opp

def measure_noise_circle_RDI(im, sci_signal_i, sci_signal_j,
                             sci_signal_i_opp, sci_signal_j_opp, 
                             aperture, ap_sz):
    ## define noise region
    nr_circle_sci = region_circle(im, sci_signal_i, sci_signal_j, aperture, ap_sz)
    nr_circle_sci_opp = region_circle(im, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, opposite=True)
    
    ## measure noise
    counts_per_ap_nr_circle_sci, ap_coords_nr_circle_sci = sum_apertures_in_region(nr_circle_sci, aperture, ap_sz)
    counts_per_ap_nr_circle_sci_opp, ap_coords_nr_circle_sci_opp = sum_apertures_in_region(nr_circle_sci_opp, aperture, ap_sz)

    #### total noise counts
    tot_noise_counts_circle = np.concatenate((counts_per_ap_nr_circle_sci, counts_per_ap_nr_circle_sci_opp))
    

    #### sigma clip 
    tot_noise_counts_circle_sgcl = sigma_clip(tot_noise_counts_circle)
    
    measured_noise_circle = np.nanstd(tot_noise_counts_circle_sgcl, ddof=1)
    
    #print("Apertures sampled:", len(tot_noise_counts_circle))
    #print("Measured noise circle", measured_noise_circle)
    
    return measured_noise_circle, nr_circle_sci, nr_circle_sci_opp
    
    
def measure_noise_ring(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                         sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                         aperture, ap_sz):
    
    ## define noise region
    nr_ring = region_ring(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz)
    
    ## measure noise
    counts_per_ap_nr_ring, ap_coords_nr_ring = sum_apertures_in_region(nr_ring, aperture, ap_sz)
    
    #### make sure that the number of apertures is even
    if len(counts_per_ap_nr_ring) % 2 == 0:
        pass
    else:
        counts_per_ap_nr_ring = counts_per_ap_nr_ring[:-1]
        ap_coords_nr_ring = ap_coords_nr_ring[:-1]
        
    
    tot_noise_counts_ring = counts_per_ap_nr_ring[0::2] + -1*counts_per_ap_nr_ring[1::2]
    
    # sigma clip
    tot_noise_counts_ring_sgcl = sigma_clip(tot_noise_counts_ring)
    
    measured_noise_ring = np.nanstd(tot_noise_counts_ring_sgcl, ddof=1)
    #print("Apertures sampled:", len(tot_noise_counts_ring))
    #print("Measured noise ring", measured_noise_ring)
    
    return measured_noise_ring, nr_ring

def measure_noise_wedge_ADI(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                         sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                         aperture, ap_sz, rotation_map, tele, disk_region, nr_angle=None, region_len_r=None, corrections=True, verbose=False):
    
    if nr_angle is None:
        # fix the nr angle such that we sample 30 apertures
        if tele == "LUVA":
            if disk_region == "struct":
                nr_angle = 22
            elif disk_region == "smooth":
                nr_angle = 11.5
        elif tele == "LUVB":
            if disk_region == "struct":
                nr_angle = 32.
            elif disk_region == "smooth":
                nr_angle = 15.5
    else:
        #print("NR angle:", nr_angle)
        pass
        
    ## define noise region
    ### wedge  region
    nr_wedge_sci = region_wedge(im, sci_signal_i, sci_signal_j, aperture, ap_sz, rotation_map, nr_angle, tele, disk_region, region_len_r=region_len_r)
    nr_wedge_sci_opp = get_opposite_wedge_region(nr_wedge_sci, im, sci_signal_i_opp, sci_signal_j_opp, ap_sz)
    nr_wedge_ref = region_wedge(im, ref_signal_i, ref_signal_j, aperture, ap_sz, rotation_map, nr_angle, tele, disk_region, region_len_r=region_len_r)
    nr_wedge_ref_opp = get_opposite_wedge_region(nr_wedge_ref, im, ref_signal_i_opp, ref_signal_j_opp, ap_sz)

    if corrections:
        #### do an r^2 correction on the wedge  region
        nr_wedge_sci = r2_correction(nr_wedge_sci)
        nr_wedge_sci_opp = r2_correction(nr_wedge_sci_opp)
        nr_wedge_ref = r2_correction(nr_wedge_ref)
        nr_wedge_ref_opp = r2_correction(nr_wedge_ref_opp)
        
        #### do an azimuthal correction on the wedge  region
        nr_wedge_sci = az_correction(nr_wedge_sci, rotation_map)
        nr_wedge_sci_opp = az_correction(nr_wedge_sci_opp, rotation_map)
        nr_wedge_ref = az_correction(nr_wedge_ref, rotation_map)
        nr_wedge_ref_opp = az_correction(nr_wedge_ref_opp, rotation_map)
    
    ## measure noise
    counts_per_ap_nr_wedge_sci, ap_coords_nr_wedge_sci = sum_apertures_in_region(nr_wedge_sci, aperture, ap_sz)
    counts_per_ap_nr_wedge_sci_opp, ap_coords_nr_wedge_sci_opp = sum_apertures_in_region(nr_wedge_sci_opp, aperture, ap_sz)
    counts_per_ap_nr_wedge_ref, ap_coords_nr_wedge_ref = sum_apertures_in_region(nr_wedge_ref, aperture, ap_sz)
    counts_per_ap_nr_wedge_ref_opp, ap_coords_nr_wedge_ref_opp = sum_apertures_in_region(nr_wedge_ref_opp, aperture, ap_sz)
    
    
    tot_sci_ap_counts_wedge = np.concatenate((counts_per_ap_nr_wedge_sci, counts_per_ap_nr_wedge_sci_opp))
    tot_ref_ap_counts_wedge = np.concatenate((counts_per_ap_nr_wedge_ref, counts_per_ap_nr_wedge_ref_opp))
    
    
    
    ##### check if sci and ref regions have equal number of apertures sampled
    if len(tot_sci_ap_counts_wedge) == len(tot_ref_ap_counts_wedge):
        pass
    elif len(tot_sci_ap_counts_wedge) > len(tot_ref_ap_counts_wedge):
        num_inds_to_cut = len(tot_sci_ap_counts_wedge) - len(tot_ref_ap_counts_wedge)
        tot_sci_ap_counts_wedge = tot_sci_ap_counts_wedge[:-num_inds_to_cut]
    elif len(tot_sci_ap_counts_wedge) < len(tot_ref_ap_counts_wedge):
        num_inds_to_cut = len(tot_ref_ap_counts_wedge) - len(tot_sci_ap_counts_wedge)
        tot_ref_ap_counts_wedge = tot_ref_ap_counts_wedge[:-num_inds_to_cut]
    else:
        assert False, "Something is wrong"
    
    tot_noise_counts_wedge = tot_sci_ap_counts_wedge + -1 * tot_ref_ap_counts_wedge
    #tot_noise_counts_wedge = np.concatenate((tot_sci_ap_counts_wedge, -1*tot_ref_ap_counts_wedge))
    #sigma clip
    tot_noise_counts_wedge_sgcl = sigma_clip(tot_noise_counts_wedge)
    measured_noise_wedge = np.nanstd(tot_noise_counts_wedge, ddof=1)
    
    if verbose:
        print("Apertures sampled:", len(tot_noise_counts_wedge))
    #print("Measured noise wedge", measured_noise_wedge)
    
    return measured_noise_wedge, nr_wedge_sci, nr_wedge_sci_opp, nr_wedge_ref, nr_wedge_ref_opp

def rotate_region(region_sci, im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz, roll_angle):
    
    region_ref = ~np.isnan(region_sci)
    region_ref = region_ref.astype(float)
    region_ref[sci_signal_i-ap_sz:sci_signal_i+ap_sz+1, sci_signal_j-ap_sz:sci_signal_j+ap_sz+1] = 1
    
    region_ref = rotate(region_ref, -roll_angle, order=0, reshape=False)
    
    region_ref[ref_signal_i-ap_sz:ref_signal_i+ap_sz+1, ref_signal_j-ap_sz:ref_signal_j+ap_sz+1] = ~aperture
    
    zero_inds = np.where(region_ref == 0)
    region_ref[zero_inds] = np.nan
    
    region_ref = region_ref * im
    
    return region_ref
    
    

def measure_noise_dynasquare_ADI(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                 sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                 aperture, ap_sz, width, height, roll_angle, corrections=True, verbose=False):
    
        
    ## define noise region
    ### dynasquare  region
    nr_dynasquare_sci = region_dynasquare(im, sci_signal_i, sci_signal_j, aperture, ap_sz, width, height)
    nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz, roll_angle)
    
    #nr_dynasquare_sci_opp = get_opposite_dynasquare_region(nr_dynasquare_sci, im, sci_signal_i_opp, sci_signal_j_opp, ap_sz)
    #nr_dynasquare_ref_opp = get_opposite_dynasquare_region(nr_dynasquare_ref, im, ref_signal_i_opp, ref_signal_j_opp, ap_sz)
    nr_dynasquare_sci_opp = np.nan * np.ones_like(nr_dynasquare_sci)
    nr_dynasquare_ref_opp = np.nan * np.ones_like(nr_dynasquare_ref)
    if corrections:
        #### do an r^2 correction on the wedge  region
        nr_dynasquare_sci = r2_correction(nr_dynasquare_sci)
        #nr_wedge_sci_opp = r2_correction(nr_wedge_sci_opp)
        nr_dynasquare_ref = r2_correction(nr_dynasquare_ref)
        #nr_wedge_ref_opp = r2_correction(nr_wedge_ref_opp)
        
# =============================================================================
#         #### do an azimuthal correction on the wedge  region
#         nr_wedge_sci = az_correction(nr_wedge_sci, rotation_map)
#         nr_wedge_sci_opp = az_correction(nr_wedge_sci_opp, rotation_map)
#         nr_wedge_ref = az_correction(nr_wedge_ref, rotation_map)
#         nr_wedge_ref_opp = az_correction(nr_wedge_ref_opp, rotation_map)
# =============================================================================
    
    ## measure noise
    counts_per_ap_nr_dynasquare_sci, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    #counts_per_ap_nr_dynasquare_sci_opp, ap_coords_nr_dynasquare_sci_opp = sum_apertures_in_region(nr_dynasquare_sci_opp, aperture, ap_sz)
    counts_per_ap_nr_dynasquare_ref, ap_coords_nr_dynasquare_ref = sum_apertures_in_region(nr_dynasquare_ref, aperture, ap_sz)
    #counts_per_ap_nr_dynasquare_ref_opp, ap_coords_nr_dynasquare_ref_opp = sum_apertures_in_region(nr_dynasquare_ref_opp, aperture, ap_sz)
    
    
    #tot_sci_ap_counts_dynasquare = np.concatenate((counts_per_ap_nr_dynasquare_sci, counts_per_ap_nr_dynasquare_sci_opp))
    #tot_ref_ap_counts_dynasquare = np.concatenate((counts_per_ap_nr_dynasquare_ref, counts_per_ap_nr_dynasquare_ref_opp))
    
    tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci
    tot_ref_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_ref
    
    
    
    ##### check if sci and ref regions have equal number of apertures sampled
    if len(tot_sci_ap_counts_dynasquare) == len(tot_ref_ap_counts_dynasquare):
        pass
    elif len(tot_sci_ap_counts_dynasquare) > len(tot_ref_ap_counts_dynasquare):
        num_inds_to_cut = len(tot_sci_ap_counts_dynasquare) - len(tot_ref_ap_counts_dynasquare)
        tot_sci_ap_counts_dynasquare = tot_sci_ap_counts_dynasquare[:-num_inds_to_cut]
    elif len(tot_sci_ap_counts_dynasquare) < len(tot_ref_ap_counts_dynasquare):
        num_inds_to_cut = len(tot_ref_ap_counts_dynasquare) - len(tot_sci_ap_counts_dynasquare)
        tot_ref_ap_counts_dynasquare = tot_ref_ap_counts_dynasquare[:-num_inds_to_cut]
    else:
        assert False, "Something is wrong"
    
    
    tot_noise_counts_dynasquare = tot_sci_ap_counts_dynasquare + -1 * tot_ref_ap_counts_dynasquare
    #tot_noise_counts_dynasquare = np.abs(tot_sci_ap_counts_dynasquare) +  np.abs(tot_ref_ap_counts_dynasquare)
    #tot_noise_counts_wedge = np.concatenate((tot_sci_ap_counts_wedge, -1*tot_ref_ap_counts_wedge))
    #sigma clip
    tot_noise_counts_dynasquare_sgcl = sigma_clip(tot_noise_counts_dynasquare)
    measured_noise_dynasquare = np.nanstd(tot_noise_counts_dynasquare, ddof=1)
    
# =============================================================================
#     if len(tot_noise_counts_dynasquare) > 0:
#         print(len(tot_noise_counts_dynasquare))
#         print(tot_sci_ap_counts_dynasquare, tot_ref_ap_counts_dynasquare)
#         print(tot_noise_counts_dynasquare)
#         print(np.std(tot_noise_counts_dynasquare, ddof=1))
#         assert False
# =============================================================================
    
    if verbose:
        print("Apertures sampled:", len(tot_noise_counts_dynasquare))

    #print("Measured noise wedge", measured_noise_wedge)
    
    return measured_noise_dynasquare, nr_dynasquare_sci, nr_dynasquare_sci_opp, nr_dynasquare_ref, nr_dynasquare_ref_opp, len(tot_noise_counts_dynasquare)

def measure_noise_dynasquare_RDI(im, sci_signal_i, sci_signal_j,
                                 sci_signal_i_opp, sci_signal_j_opp,
                                 aperture, ap_sz, width, height,  corrections=True, verbose=False):
    
        
    ## define noise region
    ### dynasquare  region
    nr_dynasquare_sci = region_dynasquare(im, sci_signal_i, sci_signal_j, aperture, ap_sz, width, height)
    #nr_dynasquare_sci_opp = get_opposite_dynasquare_region(nr_dynasquare_sci, im, sci_signal_i_opp, sci_signal_j_opp, ap_sz)
    nr_dynasquare_sci_opp = np.nan * np.ones_like(nr_dynasquare_sci)
# =============================================================================
#     if corrections:
#         #### do an r^2 correction on the wedge  region
#         nr_wedge_sci = r2_correction(nr_wedge_sci)
#         nr_wedge_sci_opp = r2_correction(nr_wedge_sci_opp)
#         nr_wedge_ref = r2_correction(nr_wedge_ref)
#         nr_wedge_ref_opp = r2_correction(nr_wedge_ref_opp)
#         
#         #### do an azimuthal correction on the wedge  region
#         nr_wedge_sci = az_correction(nr_wedge_sci, rotation_map)
#         nr_wedge_sci_opp = az_correction(nr_wedge_sci_opp, rotation_map)
#         nr_wedge_ref = az_correction(nr_wedge_ref, rotation_map)
#         nr_wedge_ref_opp = az_correction(nr_wedge_ref_opp, rotation_map)
# =============================================================================
    
    ## measure noise
    counts_per_ap_nr_dynasquare_sci, ap_coords_nr_dynasquare_sci = sum_apertures_in_region(nr_dynasquare_sci, aperture, ap_sz)
    #counts_per_ap_nr_dynasquare_sci_opp, ap_coords_nr_dynasquare_sci_opp = sum_apertures_in_region(nr_dynasquare_sci_opp, aperture, ap_sz)
    
    
    #tot_sci_ap_counts_dynasquare = np.concatenate((counts_per_ap_nr_dynasquare_sci, counts_per_ap_nr_dynasquare_sci_opp))
    tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci

    
    
    
    
    tot_noise_counts_dynasquare = tot_sci_ap_counts_dynasquare 
    #tot_noise_counts_wedge = np.concatenate((tot_sci_ap_counts_wedge, -1*tot_ref_ap_counts_wedge))
    #sigma clip
    tot_noise_counts_dynasquare_sgcl = sigma_clip(tot_noise_counts_dynasquare)
    measured_noise_dynasquare = np.nanstd(tot_noise_counts_dynasquare, ddof=1)
    
    if verbose:
        print("Apertures sampled:", len(tot_noise_counts_dynasquare))
    #print("Measured noise wedge", measured_noise_wedge)
    
    return measured_noise_dynasquare, nr_dynasquare_sci, nr_dynasquare_sci_opp, len(tot_noise_counts_dynasquare)

def measure_noise_wedge_RDI(im, sci_signal_i, sci_signal_j,
                         sci_signal_i_opp, sci_signal_j_opp,
                         aperture, ap_sz, rotation_map, tele, disk_region, corrections=True):
    
    # fix the nr angle such that we sample 30 apertures
    if tele == "LUVA":
        if disk_region == "struct":
            nr_angle = 28.
        elif disk_region == "smooth":
            nr_angle = 15.9
    elif tele == "LUVB":
        if disk_region == "struct":
            nr_angle = 40.
        elif disk_region == "smooth":
            nr_angle = 20.5
    
    ## define noise region
    ### wedge  region
    nr_wedge_sci = region_wedge(im, sci_signal_i, sci_signal_j, aperture, ap_sz, rotation_map, nr_angle, tele, disk_region)
    nr_wedge_sci_opp = get_opposite_wedge_region(nr_wedge_sci, im, sci_signal_i_opp, sci_signal_j_opp, ap_sz)

    if corrections:
        #### do an r^2 correction on the wedge  region
        nr_wedge_sci = r2_correction(nr_wedge_sci)
        nr_wedge_sci_opp = r2_correction(nr_wedge_sci_opp)
        
        #### do an azimuthal correction on the wedge  region
        nr_wedge_sci = az_correction(nr_wedge_sci, rotation_map)
        nr_wedge_sci_opp = az_correction(nr_wedge_sci_opp, rotation_map)
    
    ## measure noise
    counts_per_ap_nr_wedge_sci, ap_coords_nr_wedge_sci = sum_apertures_in_region(nr_wedge_sci, aperture, ap_sz)
    counts_per_ap_nr_wedge_sci_opp, ap_coords_nr_wedge_sci_opp = sum_apertures_in_region(nr_wedge_sci_opp, aperture, ap_sz)
    
    
    tot_noise_counts_wedge = np.concatenate((counts_per_ap_nr_wedge_sci, counts_per_ap_nr_wedge_sci_opp))
    
    
    #sigma clip
    tot_noise_counts_wedge_sgcl = sigma_clip(tot_noise_counts_wedge)
    measured_noise_wedge = np.nanstd(tot_noise_counts_wedge, ddof=1)
    
    print("Apertures sampled:", len(tot_noise_counts_wedge))
    #print("Measured noise wedge", measured_noise_wedge)
    
    return measured_noise_wedge, nr_wedge_sci, nr_wedge_sci_opp

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
    
    #cc_noise_sgcl = sigma_clip(cc_noise_vals)
    
    cc_noise = np.nanstd(cc_noise_vals, ddof=1)
    #print(cc_noise_sgcl)
    
    cc_SNR = np.abs(cc_sig) / cc_noise
    
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
    
    cc_noise_sgcl = sigma_clip(cc_noise_vals)
    
    cc_noise = np.nanstd(cc_noise_sgcl, ddof=1)
    #print(cc_noise_sgcl)
    
    cc_SNR = np.abs(cc_sig) / cc_noise
    
    return cc_SNR

def measure_signal_ADI(sub_im, noise_map_sci, noise_map_ref, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture):
    imsz, imsz = sub_im.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    sci_signal_mask = np.zeros_like(sub_im, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    ref_signal_mask = np.zeros_like(sub_im, dtype=bool)
    ref_signal_mask[ref_signal_i-ap_rad:ref_signal_i+ap_rad+1, ref_signal_j-ap_rad: ref_signal_j+ap_rad+1] = aperture
    
    sci_noise_vals = sub_im*noise_map_sci
    zero_inds = np.where(sci_noise_vals == 0)
    sci_noise_vals[zero_inds] = np.nan
    
    ref_noise_vals = sub_im*noise_map_ref
    zero_inds = np.where(ref_noise_vals == 0)
    ref_noise_vals[zero_inds] = np.nan


    sci_nr_median = np.nanmedian(sci_noise_vals)
# =============================================================================
#     plt.imshow(sci_noise_vals)
#     assert False
# =============================================================================
    ref_nr_median = np.nanmedian(ref_noise_vals)
    

    
    sci_sig = sub_im[sci_signal_mask]
    ref_sig = sub_im[ref_signal_mask]
    
    # subtract off background
    sci_sig -= sci_nr_median
    ref_sig -= ref_nr_median
    tot_sig = np.sum(sci_sig) + -1*np.sum(ref_sig)
    return np.abs(tot_sig)


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
    #print("SNR ttest", SNR_ttest)
    
    signal_ttest = (x1 - x2)
    noise_ttest = (s12*np.sqrt(1/n1 + 1/n2))
    
    return signal_ttest, noise_ttest
    
    
    


def calc_SNR_ttest_ADI(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                       aperture, ap_sz, width, height, roll_angle, corrections=True, verbose=False):
    
    

    
    imsz, imsz = im.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    sci_signal_mask = np.zeros_like(im, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    ref_signal_mask = np.zeros_like(im, dtype=bool)
    ref_signal_mask[ref_signal_i-ap_rad:ref_signal_i+ap_rad+1, ref_signal_j-ap_rad: ref_signal_j+ap_rad+1] = aperture
    
    sci_sig = im[sci_signal_mask]
    ref_sig = im[ref_signal_mask]
    
    signal_apertures = np.sum(sci_sig) + -1*np.sum(ref_sig)
    
    
    
    
    ## define noise region
    nr_dynasquare_sci = region_dynasquare(im, sci_signal_i, sci_signal_j, aperture, ap_sz, width, height)
    nr_dynasquare_ref = rotate_region(nr_dynasquare_sci, im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz, roll_angle)
    
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
    
        
    tot_sci_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_sci - np.median(counts_per_ap_nr_dynasquare_sci)
    tot_ref_ap_counts_dynasquare = counts_per_ap_nr_dynasquare_ref - np.median(counts_per_ap_nr_dynasquare_ref)
    
    
    
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
    
    total_noise = np.sqrt(noise_ttest**2 + signal_apertures)
    
    SNR_total = signal_ttest / total_noise
    
    
    noise_map_sci = ~np.isnan(nr_dynasquare_sci) 
    
    return SNR_total, SNR_classic, total_noise, noise_map_sci

def calc_SNR_ttest_RDI(im, sci_signal_i, sci_signal_j, sci_signal_i_opp, sci_signal_j_opp, 
                       aperture, ap_sz, width, height, roll_angle, corrections=True, verbose=False):
    
    imsz, imsz = im.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    sci_signal_mask = np.zeros_like(im, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    
    sci_sig = im[sci_signal_mask]
    
    signal_apertures = np.sum(sci_sig) 
    
# =============================================================================
#     print(signal_apertures)
#     assert False
# =============================================================================
    
    
    ## define noise region
    nr_dynasquare_sci = region_dynasquare(im, sci_signal_i_opp, sci_signal_j_opp, aperture, ap_sz, width, height, opposite=True)
    
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
    #print(SNR_total, signal_ttest, total_noise)

    
    noise_map_sci = ~np.isnan(nr_dynasquare_sci) 
    
    return SNR_total, SNR_classic, total_noise, noise_map_sci


    
    
    
    
    
    


def measure_signal_RDI(sub_im, noise_map_sci, sci_signal_i, sci_signal_j, aperture):
    imsz, imsz = sub_im.shape
    apsz, apsz = aperture.shape
    ap_rad = int((apsz - 1)/2)
    
    sci_signal_mask = np.zeros_like(sub_im, dtype=bool)
    sci_signal_mask[sci_signal_i-ap_rad:sci_signal_i+ap_rad+1, sci_signal_j-ap_rad: sci_signal_j+ap_rad+1] = aperture
    
    sci_noise_vals = sub_im*noise_map_sci
    zero_inds = np.where(sci_noise_vals == 0)
    sci_noise_vals[zero_inds] = np.nan


    sci_nr_median = np.nanmedian(sci_noise_vals)
      
    sci_sig = sub_im[sci_signal_mask]
    
    # subtract off background
# =============================================================================
#     sci_sig -= sci_nr_median
# =============================================================================
    
    tot_sig = np.sum(sci_sig)
    
    return np.abs(tot_sig)




