# -*- coding: utf-8 -*-

import exozodi_functions as ezf
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt


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

def region_wedge(sub_im, signal_i, signal_j, aperture, ap_sz, rotation_map, angle, opposite=False):
    
    imsz, imsz = sub_im.shape
    imctr = (imsz-1)/2

    
    noise_mask = np.zeros_like(sub_im)
    
    sig_rot = rotation_map[signal_i, signal_j]
    
    lower_ang = sig_rot-angle/2
    upper_ang = sig_rot+angle/2
    
    
    sig_dist = np.sqrt((signal_i-imctr)**2 + (signal_j-imctr)**2) 
    
    lower_dist = sig_dist - (ap_sz*2+1)
    upper_dist = sig_dist + (ap_sz*2+1)
    
    
    for i in range(imsz):
        for j in range(imsz):
            ang = rotation_map[i,j]
            dist = np.sqrt((i-imctr)**2 + (j-imctr)**2)
            if ang >= lower_ang and ang <= upper_ang and dist >= lower_dist and dist < upper_dist:
                noise_mask[i, j] = 1
    
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
    

def sum_apertures_in_region(noise_region, aperture, ap_sz):
    
    nan_map = ~np.isnan(noise_region)
    
    noise_region_median = np.nanmedian(noise_region)
    
    noise_region_bkgr_rm = noise_region - noise_region_median
    
    
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
    inds_higher = np.where(arr > np.median(arr) + thresh*np.std(arr))
    inds_lower = np.where(arr < np.median(arr) - thresh*np.std(arr))
    arr[inds_higher] = np.nan
    arr[inds_lower] = np.nan
    
    return arr
 


def r2_correction(noise_region, sig_i, sig_j):
    
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
    sig_dist = np.sqrt((sig_i-imctr)**2 + (sig_j-imctr)**2) 
    
    
    dists = np.array(dists)

    coeffs = np.polyfit(dists, vals, 2)
    x_arr = np.arange(np.min(dists), np.max(dists), 0.01)
    y_fit = coeffs[2] + coeffs[1]*x_arr + coeffs[0]*x_arr**2 
# =============================================================================
#     plt.figure()
#     plt.scatter(dists, vals)
#     plt.axvline(sig_dist, color="k")
#     plt.plot(x_arr, y_fit, color="C1")
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

def get_planet_locations_and_info(roll_angle, planet_pos_mas, pix_radius):
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


    aperture = ezf.get_psf_stamp(np.copy(sci_im), sci_signal_i, sci_signal_j, pix_radius) > 0


    ref_x = loc_of_planet_pix * np.sin(np.deg2rad(roll_angle))
    ref_y = loc_of_planet_pix * np.cos(np.deg2rad(roll_angle))

    ref_signal_i = round(ref_x + central_pixel)
    ref_signal_j = round(ref_y + central_pixel)
    
    return sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, central_pixel, aperture, imsc, diam


def measure_noise(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz, noise_region_radius=8):
    
    # measure noise before hipass
    sci_noise_region = ezf.calculate_noise_region_plan_an(sub_im, sci_signal_i, sci_signal_j, inner_r=ap_sz, outer_r=noise_region_radius)
    ref_noise_region = ezf.calculate_noise_region_plan_an(sub_im, ref_signal_i, ref_signal_j, inner_r=ap_sz, outer_r=noise_region_radius)
    
    
    sci_nr_map = ~np.isnan(sci_noise_region)
    ref_nr_map = ~np.isnan(ref_noise_region)
    nr_map = sci_nr_map | ref_nr_map
    

    noise_region = np.copy(sub_im) * nr_map
    zero_inds = np.where(nr_map == 0)
    noise_region[zero_inds] = np.nan
    
    noise_region_median = np.nanmedian(noise_region)
    
    noise_region_bkgr_rm = noise_region - noise_region_median
    
    
    measured_noise = ezf.calc_noise_in_region_two_apertures(noise_region_bkgr_rm, aperture, ap_sz)
    
    return measured_noise


def measure_noise_circle(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
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
    
    measured_noise_circle = np.nanstd(tot_noise_counts_circle_sgcl)
    
    #print("Apertures sampled:", len(tot_noise_counts_circle))
    #print("Measured noise circle", measured_noise_circle)
    
    return measured_noise_circle
    
    
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
    
    measured_noise_ring = np.nanstd(tot_noise_counts_ring_sgcl)
    #print("Apertures sampled:", len(tot_noise_counts_ring))
    #print("Measured noise ring", measured_noise_ring)
    
    return measured_noise_ring

def measure_noise_wedge(im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                         sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                         aperture, ap_sz, rotation_map):
    ## define noise region
    ### wedge  region
    nr_wedge_sci = region_wedge(im, sci_signal_i, sci_signal_j, aperture, ap_sz, rotation_map, 25.)
    nr_wedge_sci_opp = get_opposite_wedge_region(nr_wedge_sci, im, sci_signal_i_opp, sci_signal_j_opp, ap_sz)
    nr_wedge_ref = region_wedge(im, ref_signal_i, ref_signal_j, aperture, ap_sz, rotation_map, 25.)
    nr_wedge_ref_opp = get_opposite_wedge_region(nr_wedge_ref, im, ref_signal_i_opp, ref_signal_j_opp, ap_sz)

    #### do an r^2 correction on the wedge  region
    nr_wedge_sci = r2_correction(nr_wedge_sci, sci_signal_i, sci_signal_j)
    nr_wedge_sci_opp = r2_correction(nr_wedge_sci_opp, sci_signal_i, sci_signal_j)
    nr_wedge_ref = r2_correction(nr_wedge_ref, sci_signal_i, sci_signal_j)
    nr_wedge_ref_opp = r2_correction(nr_wedge_ref_opp, sci_signal_i, sci_signal_j)
    
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
    
    #sigma clip
    tot_noise_counts_wedge_sgcl = sigma_clip(tot_noise_counts_wedge)
    measured_noise_wedge = np.nanstd(tot_noise_counts_wedge_sgcl)
    
    print("Apertures sampled:", len(tot_noise_counts_wedge))
    print("Measured noise wedge", measured_noise_wedge)
    
    return measured_noise_wedge


# define some parameters
tele = "LUVA" # telescope
roll_angle = 90.
add_noise = True
add_star = True
planet_noise = True
uniform_disk = False
r2_disk = False
noise_region = "wedge"


matched_filter_dir = "../matched_filter_library/"
im_dir_path = "../data/LUVOIR-A_outputs/"



if tele == "LUVA":
    planet_pos_lamD = 10.5
    planet_pos_mas = 100.26761414789404
    
if tele == "LUVB":
    planet_pos_lamD = 7.757018897752577 # lam/D
    planet_pos_mas = 100.
    #matched_filter_datacube = np.load("/Users/mcurr/PACKAGES/coroSims//matched_filter_LUVB_datacube.npy")
    #matched_filter_single_datacube = np.load("Users/mcurr/PACKAGES/coroSims//matched_filter_LUVA_single_datacube.npy")
    
    
ap_sz_arr = np.arange(1, 3, 1)
filter_sz_arr = np.arange(1, 100, 1)
#filter_sz_arr = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

incl_arr = ["00", "30", "60", "90"]
zodi_arr = ["1", "5", "10", "20", "50", "100"]
longitude = "00"




configs = []
# do a uniform disk
# =============================================================================
# for ap_sz in ap_sz_arr:
#     for filter_sz in filter_sz_arr:
#         for zodis in zodi_arr:
#             configs.append([ap_sz, filter_sz, "00", zodis, "uniform"])
# =============================================================================

# set up configs
for ap_sz in ap_sz_arr:
    for filter_sz in filter_sz_arr:
        for incl in incl_arr:
            for zodis in zodi_arr:
                configs.append([ap_sz, filter_sz, incl, zodis, "model"])

import time
def process(config):
    start_time = time.time()
    
    ap_sz, filter_sz, incl, zodis, disk_type = config
    print(disk_type, ap_sz, filter_sz, incl, zodis)
    if ap_sz == 1:
        noise_region_radius = 7
    elif ap_sz == 2:
        noise_region_radius = 9
    elif ap_sz == 3:
        noise_region_radius = 11
    elif ap_sz == 4:
        noise_region_radius = 13
    elif ap_sz == 5:
        noise_region_radius = 17
    else:
        assert False
    
    # get planet locations in sci and ref images
    sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, central_pixel, aperture, imsc, diam = get_planet_locations_and_info(roll_angle, planet_pos_mas, ap_sz)
    
    # load matched filters according to aperture radius size
    matched_filter_fl = matched_filter_dir + "matched_filter_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), ap_sz)
    matched_filter_datacube = np.load(matched_filter_fl)
    
    if disk_type == "uniform":
        uniform_disk = True
    elif disk_type == "model":
        uniform_disk = False
    else:
        assert False, "disk type not recognized"
    
    im_dir = im_dir_path + "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}/".format(incl, longitude, zodis, round(roll_angle))
    
    cc_SNRs = []
    cc_SNRs_before_hipass = []
    measured_noise_before_hipass_arr = []
    measured_noise_after_hipass_arr = []
    measured_noise_before_hipass_out_arr = []
    measured_noise_after_hipass_out_arr = []
    frac_diff_med_arr = []
    frac_diff_std_arr = []
    converged = False
    convergence_counter = 0
    iterations = 0
    
    while not converged:
        
        iterations += 1
        #print(iterations)
    
        # synthesize images
        sci_im, ref_im, sci_planet_counts, ref_planet_counts, expected_noise_planet, expected_noise_outside, outside_loc = ezf.synthesize_images(im_dir, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, float(zodis), aperture,
                                           target_SNR=7, pix_radius=ap_sz,
                                           verbose=False, 
                                           add_noise=add_noise, 
                                           add_star=add_star, 
                                           planet_noise=planet_noise, 
                                           uniform_disk=uniform_disk,
                                           r2_disk=r2_disk)
        
        sci_out_i, sci_out_j, ref_out_i, ref_out_j = outside_loc
        
        # get opposite coords
        imsz, imsz = sci_im.shape
        imctr = (imsz-1)/2
        sci_signal_i_opp, sci_signal_j_opp  = get_opp_coords(sci_signal_i, sci_signal_j, imctr)
        ref_signal_i_opp, ref_signal_j_opp  = get_opp_coords(ref_signal_i, ref_signal_j, imctr)
        
        sci_out_i_opp, sci_out_j_opp = get_opp_coords(sci_out_i, sci_out_j, imctr)
        ref_out_i_opp, ref_out_j_opp = get_opp_coords(ref_out_i, ref_out_j, imctr)
       
        
        # calculate maps
        rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im, imsc, diam, IWA_lamD=7.5, plotting=False)
        
        # set non-valid inds to nan
        sci_im[~valid_mask] = np.nan
        ref_im[~valid_mask] = np.nan
        
        # perform adi 
        sub_im = sci_im - ref_im
        
        # perform high pass filter on the sub im
        sub_im_hipass = ezf.high_pass_filter(sub_im, filtersize=filter_sz)
        
        
        
        # evaluate SNR before hipass
        #sub_SNR = ezf.calculate_SNR_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=False, force_signal=None)
        
        
        # measure noise
        if noise_region == "circle":
            
            measured_noise_before_hipass = measure_noise_circle(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                                sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                                aperture, ap_sz)
            
            measured_noise_after_hipass = measure_noise_circle(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                               sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                               aperture, ap_sz)
            
            measured_noise_before_hipass_out = measure_noise_circle(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                    sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                    aperture, ap_sz)
            
            measured_noise_after_hipass_out = measure_noise_circle(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                   sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                   aperture, ap_sz)
            
        elif noise_region == "ring":
            
            measured_noise_before_hipass = measure_noise_ring(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                              sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                              aperture, ap_sz)
            
            measured_noise_after_hipass = measure_noise_ring(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                             sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                             aperture, ap_sz)
            
            measured_noise_before_hipass_out = measure_noise_ring(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                  sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                  aperture, ap_sz)
            
            measured_noise_after_hipass_out = measure_noise_ring(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                 sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                 aperture, ap_sz)
        
        elif noise_region == "wedge":
            measured_noise_before_hipass = measure_noise_wedge(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                               sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                               aperture, ap_sz, rotation_map)
            
            measured_noise_after_hipass = measure_noise_wedge(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                              sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                              aperture, ap_sz, rotation_map)
            
            measured_noise_before_hipass_out = measure_noise_wedge(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                   sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                   aperture, ap_sz, rotation_map)
            
            measured_noise_after_hipass_out = measure_noise_wedge(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                  sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                  aperture, ap_sz, rotation_map)
            
        
            
        
        measured_noise_before_hipass_arr.append(measured_noise_before_hipass)
        measured_noise_after_hipass_arr.append(measured_noise_after_hipass)
        measured_noise_before_hipass_out_arr.append(measured_noise_before_hipass_out)
        measured_noise_after_hipass_out_arr.append(measured_noise_after_hipass_out)
        
        
        # get expected noise
        expected_noise = np.sqrt(expected_noise_planet)
        expected_noise_out = np.sqrt(expected_noise_outside)

        

        # cross correlation
        
        cc_map_before_hipass = ezf.calculate_cc_map(matched_filter_datacube, sub_im, valid_mask)
        
        cc_SNR_before_hipass = ezf.cc_SNR_known_loc(cc_map_before_hipass, sci_signal_i, sci_signal_j, ap_sz, roll_angle, aperture, central_pixel, noise_region_radius, r2_correct=False, mask_antisignal=True)
        cc_SNRs_before_hipass.append(cc_SNR_before_hipass)
        
        
        cc_map = ezf.calculate_cc_map(matched_filter_datacube, sub_im_hipass, valid_mask)
        
        cc_SNR = ezf.cc_SNR_known_loc(cc_map, sci_signal_i, sci_signal_j, ap_sz, roll_angle, aperture, central_pixel, noise_region_radius, r2_correct=False, mask_antisignal=True)
        
        cc_SNRs.append(cc_SNR)
        
        if iterations > 1:
            # compute fractional difference of medians
            old_median = np.median(cc_SNRs[:-1])
            new_median = np.median(cc_SNRs)
            
            frac_diff_med = np.abs((new_median - old_median) / old_median)
            #print("med:",old_median, new_median, frac_diff_med)
            
            old_std = np.std(cc_SNRs[:-1])
            new_std = np.std(cc_SNRs)
            frac_diff_std = np.abs((new_std - old_std) / old_std)
            #print("std:", old_std, new_std, frac_diff_std)
            
            frac_diff_med_arr.append(frac_diff_med)
            frac_diff_std_arr.append(frac_diff_std)

        if iterations > 500:
            

            if (frac_diff_med < 0.01) and (frac_diff_std < 0.01):
                #print("Converged")
                convergence_counter += 1
                if convergence_counter == 10:
                    print("Converged")
                    converged = True
                    
            else:
                # reset the convergence counter
                convergence_counter = 0

        if iterations == 1000:
            print("NOT CONVERGED: Iteration limit reached.")
            break
        
    median_cc_SNR = np.median(cc_SNRs)
    median_cc_SNR_before_hipass = np.median(cc_SNRs_before_hipass)
    median_measured_noise_before_hipass = np.median(measured_noise_before_hipass_arr)
    median_measured_noise_after_hipass = np.median(measured_noise_after_hipass_arr)
    median_measured_noise_before_hipass_out = np.median(measured_noise_before_hipass_out_arr)
    median_measured_noise_after_hipass_out = np.median(measured_noise_after_hipass_out_arr)
    
    print(median_measured_noise_after_hipass_out / expected_noise_out)
    return_arr = np.array([uniform_disk, ap_sz, filter_sz, int(incl), int(zodis), median_cc_SNR, median_cc_SNR_before_hipass, iterations,
                           median_measured_noise_before_hipass, median_measured_noise_after_hipass, expected_noise,
                           median_measured_noise_before_hipass_out, median_measured_noise_after_hipass_out, expected_noise_out])
    
    end_time = time.time()
    print("Time elapsed for process: {} s".format(round(end_time - start_time, 2)))
    return return_arr


parallel = True

# sequential runs
if parallel == False:
    data = []
    for config in configs:
        
        data_arr  = process(config)
        data.append(data_arr)
        
    data = np.array(data)
    
# =============================================================================
#     header = "ap_sz filter_sz incl zodis median_cc_SNR median_cc_SNR_before_hipass iterations measured_noise_before_hipass measured_noise_after_hipass expected_noise"
#     np.savetxt("data.dat", data, header=header, comments='')
# =============================================================================
    
        

# parallel runs
elif parallel == True:
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=39)(delayed(process)(config) for config in configs)
    
    header = "uniform_disk ap_sz filter_sz incl zodis median_cc_SNR median_cc_SNR_before_hipass iterations measured_noise_before_hipass measured_noise_after_hipass expected_noise median_measured_noise_before_hipass_out median_measured_noise_after_hipass_out expected_noise_out"
    np.savetxt("data_{}.dat".format(noise_region), results, header=header, comments='')



                
                
