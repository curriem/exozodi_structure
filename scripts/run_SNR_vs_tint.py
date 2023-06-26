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
except IndexError:
    
    tele = "LUVB"
    DI = "ADI"
    print("WARNING: NO TELE, DI, NOISE REGION SPECIFIED. USING {}, {}, {}.".format(tele, DI, noise_region))

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

mode = "scatteredlight"


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
height = 3
width = 3

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

    
    SNR_after_hipass_arr = []
    
    
    measured_noise_after_hipass_arr = []
    
    

    niter = 1000
    
    for iterations in range(niter):
        
        
        if iterations % 100 == 0:
            print(iterations)
        if iterations == 0:
            syn_verbose = True
        else:
            syn_verbose = False
    
        # synthesize images
        sci_im, ref_im,  \
        expected_noise_planet, expected_noise_outside, outside_loc, tot_tint = ezf.synthesize_images_ADI3(im_dir, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, float(incl), float(zodis), aperture, roll_angle,
                                                                                               target_SNR=None, tot_tint=tot_tint, pix_radius=ap_sz,
                                                                                               verbose=syn_verbose, 
                                                                                               add_noise=add_noise, 
                                                                                               add_star=add_star, 
                                                                                               planet_noise=planet_noise, 
                                                                                               uniform_disk=uniform_disk)        
        
        


        # get opposite coords
        imsz, imsz = sci_im.shape
        imctr = (imsz-1)/2
        sci_signal_i_opp, sci_signal_j_opp  = ezf.get_opp_coords(sci_signal_i, sci_signal_j, imctr)
        
        
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
        expected_noise_out = np.sqrt(expected_noise_outside)
        
        

            
        
            
                
        SNR_HPMF, mf_sig, measured_noise = ezf.calc_SNR_HPMF_ADI(sub_im_hipass, matched_filter_datacube, matched_filter_datacube_single,
                                                                 sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, 
                                                                 aperture, ap_sz, width, height, roll_angle)
        

        
        if mf_sig < 0:
            mf_sig = 0
            SNR_HPMF = 0
        
       
        
        SNR_after_hipass_arr.append(SNR_HPMF)
        
        measured_noise_after_hipass_arr.append(measured_noise)

            
            
        
        
        

    
        
    median_SNR_after_hipass = np.nanmedian(SNR_after_hipass_arr)
    median_measured_noise_after_hipass = np.nanmedian(measured_noise_after_hipass_arr)
    
    
    
    verbose = False
    if verbose:
        print("Median SNR after hipass:", median_SNR_after_hipass)
        print("Expected noise:", expected_noise)
        print("Expected noise out:", expected_noise_out)
        print("Median measured/expected noise after hipass:", median_measured_noise_after_hipass/expected_noise)

        
        plt.figure()
        plt.hist(SNR_after_hipass_arr, bins=30)
        plt.axvline(np.nanmedian(SNR_after_hipass_arr), color="k")
        plt.axvline(np.nanmean(SNR_after_hipass_arr), color="k", linestyle=":")
        plt.title("Median: {}, Mean: {}".format(np.nanmedian(SNR_after_hipass_arr), np.nanmean(SNR_after_hipass_arr)))
        plt.show()
        

    return_arr = np.array([im_sz/filter_sz, int(incl), int(zodis), 
                           median_SNR_after_hipass,
                           median_measured_noise_after_hipass, expected_noise,
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
    
    header = "filter_sz_pix incl zodis median_SNR_after_hipass measured_noise_after_hipass expected_noise tot_tint"
    save_fl = "SNR_vs_tot_tint_{}_{}".format(tele, DI)
    save_fl += "_{}".format(mode)
    save_fl += ".dat"
    np.savetxt(save_fl, results, header=header, comments='')
 

