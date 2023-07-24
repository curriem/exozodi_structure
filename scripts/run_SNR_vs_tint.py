# -*- coding: utf-8 -*-

import exozodi_functions as ezf
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

# define some parameters
roll_angle = 90.
add_noise = True
add_star = True
planet_noise = True


try:
    tele = str(sys.argv[1])
    DI = str(sys.argv[2])
    mode = str(sys.argv[3])
except IndexError:
    
    tele = "LUVB"
    DI = "ADI"
    mode = "scatteredlight"
    print("WARNING: NO TELE, DI, NOISE REGION SPECIFIED. USING {}, {}.".format(tele, DI))


matched_filter_dir = "../matched_filter_library/"


    

if tele == "LUVA":
    planet_pos_lamD = 10.5
    planet_pos_mas = 100.26761414789404
    
    im_dir_path = "../data/LUVOIR-A_outputs/"
    IWA = 7.5
    OWA = 40#22.
    
if tele == "LUVB":
    planet_pos_lamD = 7.0 # lam/D
    planet_pos_mas = 100.26761414789404
    
    im_dir_path = "../data/LUVOIR-B_outputs/"
    im_dir_path_od = "../data/LUVOIR-B_outputs_opticaldepth/"
    IWA = 2.5
    OWA = 22 #13

    
    
filter_sz_arr_pix = np.arange(2, 51, 1)
im_sz = 101

incl_arr = ["00", "30", "60", "90"]
zodi_arr = ["1", "5", "10", "20", "50", "100"]
longitude = "00"


#tot_tint_arr = np.logspace(2, 10, 100)
tot_tint_arr = np.logspace(2, 8, 25)


configs = []

df_fl = 'data_{}_{}'.format(tele, DI)
df_fl += ".dat"
df = pd.read_csv(df_fl, sep=" ", header=0)

# set up configs
for filter_sz in filter_sz_arr_pix:
    for tot_tint in tot_tint_arr:
        for incl in incl_arr:
            for zodis in zodi_arr:
                configs.append([1, im_sz / filter_sz, incl, zodis, "model", tot_tint])



# define height and width of noise region:
noise_region = "planet"
inner_r = 2
outer_r = 6


import time
def process(config):
    start_time = time.time()
    
    ap_sz, filter_sz, incl, zodis, disk_type, tot_tint = config
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
    
    if mode == "scatteredlight":
        im_dir = im_dir_path + "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}/".format(incl, longitude, zodis, round(roll_angle))
        im_dir += "/"
    elif mode == "opticaldepth":
        im_dir = im_dir_path_od + "opticaldepth-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}/".format(incl, longitude, zodis, round(roll_angle))
        im_dir += "/"

    
    SNR_HPMF_arr = []
    SNR_HPAP_arr = []
    measured_noise_HPMF_arr = []
    measured_noise_HPAP_arr = []
    
    

    niter = 1000
    
    for iterations in range(niter):
        
        
        if iterations % 100 == 0:
            print(iterations)
        if iterations == 0:
            syn_verbose = True
        else:
            syn_verbose = False
    
        # synthesize images
        sci_im_ap, ref_im_ap, sci_im_mf, ref_im_mf,  \
            expected_noise_planet_ap, expected_noise_planet_mf, \
            expected_noise_bkgr_ap, expected_noise_bkgr_mf, \
            outside_loc, \
            tot_tint_ap, tot_tint_mf, \
            sub_disk_im_noiseless = ezf.synthesize_images_ADI3(im_dir, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, float(incl), float(zodis), aperture, roll_angle,
                                                                                               target_SNR=None, tot_tint=tot_tint, pix_radius=ap_sz,
                                                                                               verbose=syn_verbose, 
                                                                                               add_noise=add_noise, 
                                                                                               add_star=add_star, 
                                                                                               planet_noise=planet_noise, 
                                                                                               uniform_disk=uniform_disk,
                                                                                               background="region",
                                                                                               simple_planet=False,
                                                                                               matched_filter_datacube_single=matched_filter_datacube_single)

        # get opposite coords
        imsz, imsz = sci_im_mf.shape
        imctr = (imsz-1)/2
        
        
        # calculate maps
        rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im_mf, imsc, diam, IWA_lamD=IWA, OWA_lamD=OWA, plotting=False)
        
        # set non-valid inds to nan
        sci_im_ap[~valid_mask] = np.nan
        ref_im_ap[~valid_mask] = np.nan
        sci_im_mf[~valid_mask] = np.nan
        ref_im_mf[~valid_mask] = np.nan
        
        
        
        # perform subtraction 
        sub_im_ap = sci_im_ap - ref_im_ap
        sub_im_mf = sci_im_mf - ref_im_mf

        
        # perform high pass filter on the sub im
        sub_im_hipass_ap = ezf.high_pass_filter(sub_im_ap, filtersize=filter_sz)
        sub_im_hipass_mf = ezf.high_pass_filter(sub_im_mf, filtersize=filter_sz)


        
        # get expected noise
        expected_noise_ap = np.sqrt(expected_noise_planet_ap)
        expected_noise_bkgr_ap = np.sqrt(expected_noise_bkgr_ap)
        
        expected_noise_mf = np.sqrt(expected_noise_planet_mf)
        expected_noise_bkgr_mf = np.sqrt(expected_noise_bkgr_mf)
        
        

            
        
            
                
        SNR_HPMF, sig_HPMF, measured_noise_HPMF, measured_noise_HPMF_bkgr = ezf.calc_SNR_HPMF_ADI(sub_im_hipass_mf, matched_filter_datacube, matched_filter_datacube_single,
                                                         sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                                                         aperture, ap_sz, inner_r, outer_r, roll_angle, noise_region=noise_region)
        
        SNR_HPAP, sig_HPAP, measured_noise_HPAP, measured_noise_HPAP_bkgr = ezf.calc_SNR_HPAP_ADI(sub_im_hipass_ap,
                                                         sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                                                         aperture, ap_sz, inner_r, outer_r, roll_angle, noise_region=noise_region)
        
        if sig_HPMF < 0:
            sig_HPMF = 0
            SNR_HPMF = 0
            
        if sig_HPAP < 0:
            sig_HPAP = 0
            SNR_HPAP = 0
       
        
       
        SNR_HPMF_arr.append(SNR_HPMF)
        measured_noise_HPMF_arr.append(measured_noise_HPMF_bkgr)
        
        SNR_HPAP_arr.append(SNR_HPAP)
        measured_noise_HPAP_arr.append(measured_noise_HPAP_bkgr)
       
            
            
        
        
        

    
        
    med_meas_noise_HPMF = np.nanmedian(measured_noise_HPMF_arr)
    med_meas_noise_HPAP = np.nanmedian(measured_noise_HPAP_arr)
    
    med_SNR_HPMF = np.nanmedian(SNR_HPMF_arr)
    med_SNR_HPAP = np.nanmedian(SNR_HPAP_arr)
    
    
    
    
    verbose = False
    if verbose:
        print("Median SNR HPMF:", med_SNR_HPMF)
        print("Median SNR HPAP:", med_SNR_HPAP)

        print("Expected noise MF:", expected_noise_mf)
        print("Expected noise AP:", expected_noise_ap)

        print("Median measured/expected noise MF:", med_meas_noise_HPMF/expected_noise_bkgr_mf)
        print("Median measured/expected noise AP:", med_meas_noise_HPAP/expected_noise_bkgr_ap)
        
        
        fig, axes = plt.subplots(2,2)
        
        # SNR HPAP
        axes[0,0].hist(SNR_HPAP_arr, bins=30)
        axes[0,0].axvline(np.nanmedian(SNR_HPAP_arr), color="k")
        axes[0,0].axvline(np.nanmean(SNR_HPAP_arr), color="k", linestyle=":")
        axes[0,0].set_title("SNR HPAP, Median: {}, Mean: {}".format(round(np.nanmedian(SNR_HPAP_arr), 3), round(np.nanmean(SNR_HPAP_arr), 3)))

        # SNR HPMF
        axes[0,1].hist(SNR_HPMF_arr, bins=30)
        axes[0,1].axvline(np.nanmedian(SNR_HPMF_arr), color="k")
        axes[0,1].axvline(np.nanmean(SNR_HPMF_arr), color="k", linestyle=":")
        axes[0,1].set_title("SNR HPMF, Median: {}, Mean: {}".format(round(np.nanmedian(SNR_HPMF_arr), 3), round(np.nanmean(SNR_HPMF_arr), 3)))
        
        
        # noise HPAP
        axes[1,0].hist(measured_noise_HPAP_arr, bins=30)
        axes[1,0].axvline(np.nanmedian(measured_noise_HPAP_arr), color="k")
        axes[1,0].axvline(np.nanmean(measured_noise_HPAP_arr), color="k", linestyle=":")
        axes[1,0].axvline(expected_noise_bkgr_ap, color="red", linestyle="-", linewidth=3, alpha=0.5)
        axes[1,0].set_title("noise, True: {}, Median: {}, Mean: {}".format(round(expected_noise_bkgr_ap, 3), round(np.nanmedian(measured_noise_HPAP_arr), 3), round(np.nanmean(measured_noise_HPAP_arr), 3)))
        
        
        # noise HPMF
        axes[1,1].hist(measured_noise_HPMF_arr, bins=30)
        axes[1,1].axvline(np.nanmedian(measured_noise_HPMF_arr), color="k")
        axes[1,1].axvline(np.nanmean(measured_noise_HPMF_arr), color="k", linestyle=":")
        axes[1,1].axvline(expected_noise_bkgr_mf, color="red", linestyle="-", linewidth=3, alpha=0.5)
        axes[1,1].set_title("noise, True: {}, Median: {}, Mean: {}".format(round(expected_noise_bkgr_mf, 3), round(np.nanmedian(measured_noise_HPMF_arr), 3), round(np.nanmean(measured_noise_HPMF_arr), 3)))
        
        fig.tight_layout()
        

    return_arr = np.array([im_sz/filter_sz, int(incl), int(zodis), 
                           med_SNR_HPAP, med_SNR_HPMF, 
                           med_meas_noise_HPAP, med_meas_noise_HPMF,
                           expected_noise_ap, expected_noise_mf,
                           expected_noise_bkgr_ap, expected_noise_bkgr_mf,
                           tot_tint])
    
    end_time = time.time()
    return return_arr


parallel = True

# sequential runs
if parallel == False:
    data = []
    
    #configs = [([1, 101/10., "00", "1", "uniform"])]
    #configs = [([1, 101/5., "60", "50", "model", 1000])]
    for config in configs[:1]:
        
        data_arr  = process(config)
        data.append(data_arr)
        print(data_arr)
        
    data = np.array(data)
    
        

# parallel runs
elif parallel == True:
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=40)(delayed(process)(config) for config in configs)
    
    header = "filter_sz_pix incl zodis med_SNR_HPAP med_SNR_HPMF med_meas_noise_HPAP med_meas_noise_HPMF expected_noise_ap expected_noise_mf expected_noise_bkgr_ap expected_noise_bkgr_mf tot_tint"
    save_fl = "SNR_vs_tot_tint_{}_{}".format(tele, DI)
    save_fl += "_{}".format(mode)
    save_fl += ".dat"
    np.savetxt(save_fl, results, header=header, comments='')
 

