# -*- coding: utf-8 -*-

import exozodi_functions as ezf
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt

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
    



# define some parameters
tele = "LUVA" # telescope
roll_angle = 90.
add_noise = True
add_star = True
planet_noise = True
uniform_disk = False
r2_disk = False


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
    
    
ap_sz_arr = np.arange(1, 6, 1)
filter_sz_arr = np.arange(1, 100, 1)
#filter_sz_arr = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

incl_arr = ["00", "30", "60", "90"]
zodi_arr = ["1", "5", "10", "20", "50", "100"]
longitude = "00"




configs = []
# do a uniform disk
for ap_sz in ap_sz_arr:
    for filter_sz in filter_sz_arr:
        for zodis in zodi_arr:
            configs.append([ap_sz, filter_sz, "00", zodis, "uniform"])

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
        
       
        
        # calculate maps
        rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im, imsc, diam, IWA_lamD=7.5, plotting=False)
        
        # set non-valid inds to nan
        sci_im[~valid_mask] = np.nan
        ref_im[~valid_mask] = np.nan
        
        # perform adi 
        sub_im = sci_im - ref_im
        
# =============================================================================
#         plt.figure()
#         plt.imshow(sub_im, origin="lower")
#         plt.axhline(sci_signal_i, color="green")
#         plt.axvline(sci_signal_j, color="green")
#         plt.axhline(sci_out_i, color="red")
#         plt.axvline(sci_out_j, color="red")
#         
#         assert False
# =============================================================================

        
        # evaluate SNR before hipass
        
        
        #sub_SNR = ezf.calculate_SNR_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=False, force_signal=None)
        
        
        #measured_noise_before_hipass = measure_noise(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz, noise_region_radius=noise_region_radius)
        measured_noise_before_hipass, _, _ = ezf.calc_noise_in_region_testing(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz)
        measured_noise_before_hipass_arr.append(measured_noise_before_hipass)
        
        
        expected_noise = np.sqrt(expected_noise_planet)
        
        cc_map_before_hipass = ezf.calculate_cc_map(matched_filter_datacube, sub_im, valid_mask)
        
        cc_SNR_before_hipass = ezf.cc_SNR_known_loc(cc_map_before_hipass, sci_signal_i, sci_signal_j, ap_sz, roll_angle, aperture, central_pixel, noise_region_radius, r2_correct=False, mask_antisignal=True)
        cc_SNRs_before_hipass.append(cc_SNR_before_hipass)
        
        
        # perform high pass filter on the sub im
        sub_im_hipass = ezf.high_pass_filter(sub_im, filtersize=filter_sz)
        
        
        #measured_noise_after_hipass = measure_noise(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz, noise_region_radius=noise_region_radius)
        measured_noise_after_hipass, _, _ = ezf.calc_noise_in_region_testing(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture, ap_sz)
        measured_noise_after_hipass_arr.append(measured_noise_after_hipass)
        
        
        # outside region
        #measured_noise_before_hipass_out = measure_noise(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j, aperture, ap_sz, noise_region_radius=5)
        measured_noise_before_hipass_out, _, _ = ezf.calc_noise_in_region_testing(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j, aperture, ap_sz)
        measured_noise_before_hipass_out_arr.append(measured_noise_before_hipass_out)
        
        #measured_noise_after_hipass_out = measure_noise(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j, aperture, ap_sz, noise_region_radius=5)
        measured_noise_after_hipass_out, _, _ = ezf.calc_noise_in_region_testing(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j, aperture, ap_sz)
        measured_noise_after_hipass_out_arr.append(measured_noise_after_hipass_out)
        expected_noise_out = np.sqrt(expected_noise_outside)
        
        # evaluate SNR after hipass 
        #sub_SNR_hipass = ezf.calculate_SNR_ADI(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=False, force_signal=None)
        
        
        # cross correlation
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


parallel = False

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
    np.savetxt("data.dat", results, header=header, comments='')



                
                
