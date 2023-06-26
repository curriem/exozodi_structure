# -*- coding: utf-8 -*-

import exozodi_functions as ezf
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

# define some parameters
roll_angle = 30.
add_noise = True
add_star = True
planet_noise = True


try:
    tele = str(sys.argv[1])
    DI = str(sys.argv[2])
    noise_region = str(sys.argv[3])
    planloc = str(sys.argv[4])
except IndexError:
    
    tele = "LUVB"
    DI = "ADI"
    noise_region = None
    planloc = "planin"
    print("WARNING: NO TELE, DI, NOISE REGION SPECIFIED. USING {}, {}, {}.".format(tele, DI, noise_region))

matched_filter_dir = "../matched_filter_library/"

if planloc == "planin":
    planet_outside = False
elif planloc == "planout":
    planet_outside = True
    

if tele == "LUVA":
    planet_pos_lamD = 10.5
    planet_pos_mas = 100.26761414789404
    if planloc == "planin":
        im_dir_path = "../data/LUVOIR-A_outputs/"
    elif planloc == "planout":
        im_dir_path = "../data/LUVOIR-A_outputs_smooth/"
    IWA = 7.5
    OWA = 40#22.
    
if tele == "LUVB":
    planet_pos_lamD = 7.0 # lam/D
    planet_pos_mas = 100.26761414789404
    if planloc == "planin":
        im_dir_path = "../data/LUVOIR-B_outputs/"
    if planloc == "planout":
        im_dir_path = "../data/LUVOIR-B_outputs_smooth/"
    IWA = 2.5
    OWA = 22 #13

    
    
ap_sz_arr = np.arange(1, 2, 1)
filter_sz_arr_pix = np.arange(2, 51, 1)
im_sz = 101
filter_sz_arr_fourier = im_sz / filter_sz_arr_pix
#filter_sz_arr = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

incl_arr = ["00", "30", "60", "90"]
zodi_arr = ["1", "5", "10", "20", "50", "100"]
longitude = "00"




configs = []

# do a uniform disk
for ap_sz in ap_sz_arr:
    for filter_sz in filter_sz_arr_fourier:
        for zodis in zodi_arr:
            configs.append([ap_sz, filter_sz, "00", zodis, "uniform"])

# set up configs
for ap_sz in ap_sz_arr:
    for filter_sz in filter_sz_arr_fourier:
        for incl in incl_arr:
            for zodis in zodi_arr:
                configs.append([ap_sz, filter_sz, incl, zodis, "model"])


# define height and width of noise region:
noise_region = "planet"
inner_r = None
outer_r = 3

import time
def process(config):
    start_time = time.time()
    
    ap_sz, filter_sz, incl, zodis, disk_type = config
    print(disk_type, ap_sz, filter_sz, incl, zodis)
    
    if disk_type == "uniform":
        uniform_disk = True
    elif disk_type == "model":
        uniform_disk = False
    else:
        assert False, "disk type not recognized"
        

    
    
    # get planet locations in sci and ref images
    sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, central_pixel, aperture, imsc, diam = ezf.get_planet_locations_and_info(roll_angle, planet_pos_mas, ap_sz, im_dir_path)
    
    # load matched filters according to aperture radius size
    matched_filter_fl = matched_filter_dir + "matched_filter_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), ap_sz)
    matched_filter_single_fl = matched_filter_dir + "matched_filter_single_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), ap_sz)
    matched_filter_datacube = np.load(matched_filter_fl)
    matched_filter_datacube_single = np.load(matched_filter_single_fl)
    
    
    im_dir = im_dir_path + "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}".format(incl, longitude, zodis, round(roll_angle))
    im_dir += "/"

    SNR_HPMF_arr = []
    measured_noise_after_hipass_arr = []    
    signal_arr = []
    
    

    niter = 1000
    
    for iterations in range(niter):
        
        
        if iterations % 100 == 0:
            print(iterations)
        if iterations == 0:
            syn_verbose = True
        else:
            syn_verbose = False
    
        # synthesize images
        if DI == "ADI":
            sci_im, ref_im,  \
            expected_noise_planet, expected_noise_bkgr, outside_loc, tot_tint, sub_disk_im_noiseless = ezf.synthesize_images_ADI3(im_dir, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, float(incl), float(zodis), aperture, roll_angle,
                                                                                                   target_SNR=7, pix_radius=ap_sz,
                                                                                                   verbose=syn_verbose, 
                                                                                                   add_noise=add_noise, 
                                                                                                   add_star=add_star, 
                                                                                                   planet_noise=planet_noise, 
                                                                                                   uniform_disk=uniform_disk,
                                                                                                   background="planetloc")
            sci_out_i, sci_out_j, ref_out_i, ref_out_j = outside_loc
        elif DI == "RDI":
            
            sci_im, ref_im, \
                expected_noise_planet, expected_noise_bkgr, outside_loc, tot_tint, sub_disk_im_noiseless = ezf.synthesize_images_RDI3(im_dir, sci_signal_i, sci_signal_j, float(zodis), aperture,
                                                                                                   target_SNR=7, pix_radius=ap_sz, 
                                                                                                   verbose=syn_verbose,
                                                                                                   add_noise=add_noise, 
                                                                                                   add_star=add_star, 
                                                                                                   planet_noise=planet_noise, 
                                                                                                   uniform_disk=uniform_disk,
                                                                                                   zerodisk=False,
                                                                                                   background="planetloc")
            sci_out_i, sci_out_j = outside_loc
            
            
        
        

        
        # get opposite coords
        imsz, imsz = sci_im.shape
        imctr = (imsz-1)/2
        sci_signal_i_opp, sci_signal_j_opp  = ezf.get_opp_coords(sci_signal_i, sci_signal_j, imctr)
        
        sci_out_i_opp, sci_out_j_opp = ezf.get_opp_coords(sci_out_i, sci_out_j, imctr)
        
        if DI == "ADI":
            ref_signal_i_opp, ref_signal_j_opp  = ezf.get_opp_coords(ref_signal_i, ref_signal_j, imctr)
            ref_out_i_opp, ref_out_j_opp = ezf.get_opp_coords(ref_out_i, ref_out_j, imctr)

       
        
        # calculate maps
        rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im, imsc, diam, IWA_lamD=IWA, OWA_lamD=OWA, plotting=False)
        
        # set non-valid inds to nan
        sci_im[~valid_mask] = np.nan
        ref_im[~valid_mask] = np.nan
        
        
        
        # perform subtraction 
        sub_im = sci_im - ref_im
        

        # perform high pass filter on the sub im
        sub_im_hipass = ezf.high_pass_filter(sub_im, filtersize=filter_sz)


        # get expected noise
        expected_noise = np.sqrt(expected_noise_planet)
        expected_noise_bkgr = np.sqrt(expected_noise_bkgr)

        
    
            
        
        if DI == "ADI":
           
            SNR_HPMF, mf_sig, measured_noise, measured_noise_bkgr = ezf.calc_SNR_HPMF_ADI(sub_im_hipass, matched_filter_datacube, matched_filter_datacube_single,
                                                             sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                                                             aperture, ap_sz, inner_r, outer_r, roll_angle, noise_region=noise_region)
            
        
        elif DI == "RDI":
         
            SNR_HPMF, mf_sig, measured_noise, measured_noise_bkgr = ezf.calc_SNR_HPMF_RDI(sub_im_hipass, matched_filter_datacube_single,
                                                                     sci_signal_i, sci_signal_j, 
                                                                     aperture, ap_sz, inner_r, outer_r, noise_region=noise_region)
        
        

        
        
        
        if mf_sig < 0:
            mf_sig = 0
            SNR_HPMF = 0
        
        SNR_HPMF_arr.append(SNR_HPMF)
        signal_arr.append(mf_sig)
        
        measured_noise_after_hipass_arr.append(measured_noise_bkgr)

        

        
        

            
        
        
        

    
        
    median_measured_noise_after_hipass = np.nanmedian(measured_noise_after_hipass_arr)
    median_signal = np.nanmedian(signal_arr)
    median_SNR_HPMF = np.nanmedian(SNR_HPMF_arr)
    

    std_SNR_HPMF = np.nanstd(SNR_HPMF_arr, ddof=1)
    std_noise_after_hipass = np.nanstd(measured_noise_after_hipass_arr, ddof=1)

    
    verbose = False
    if verbose:
        print("Median SNR:", median_SNR_HPMF)
        print("Median SNR after hipass:", median_SNR_HPMF)
        print("Expected noise:", expected_noise)
        print("Median measured/expected noise after hipass:", median_measured_noise_after_hipass/expected_noise)
        print("Median signal:", median_signal)
        
        
        
        plt.figure()
        plt.hist(SNR_HPMF_arr, bins=30)
        plt.axvline(np.nanmedian(SNR_HPMF_arr), color="k")
        plt.axvline(np.nanmean(SNR_HPMF_arr), color="k", linestyle=":")
        plt.title("SNR HPMF, Median: {}, Mean: {}".format(np.nanmedian(SNR_HPMF_arr), np.nanmean(SNR_HPMF_arr)))
        plt.show()
        
        plt.figure()
        plt.hist(signal_arr, bins=30)
        plt.axvline(np.nanmedian(signal_arr), color="k")
        plt.axvline(np.nanmean(signal_arr), color="k", linestyle=":")
        plt.title("signal, Median: {}, Mean: {}".format(np.nanmedian(signal_arr), np.nanmean(signal_arr)))
        plt.show()
        
        

    return_arr = np.array([uniform_disk, ap_sz, im_sz/filter_sz, int(incl), int(zodis), 
                           median_SNR_HPMF,
                           iterations,
                           median_measured_noise_after_hipass, expected_noise_bkgr,
                           std_SNR_HPMF, std_noise_after_hipass])
    
    end_time = time.time()
    return return_arr


parallel = False

# sequential runs
if parallel == False:
    data = []
    
    configs = [([1, 101/50, "60", "20", "model"])]
    #configs = [([1, 101/4., "60", "50", "model"])]
    for config in configs:
        
        data_arr  = process(config)
        data.append(data_arr)
        print(data_arr)
        
    data = np.array(data)
    

        

# parallel runs
elif parallel == True:
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=4)(delayed(process)(config) for config in configs[:3])
    
    header = "uniform_disk ap_sz filter_sz_pix incl zodis median_SNR_after_hipass iterations measured_noise_after_hipass expected_noise_bkgr std_SNR_after_hipass std_noise_after_hipass"
    save_fl = "data_{}_{}".format(tele, DI)
    if planloc == "planout":
        save_fl += "_planout"
    save_fl += ".dat"
    np.savetxt(save_fl, results, header=header, comments='')
 
