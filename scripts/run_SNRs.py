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
r2_disk = False


try:
    tele = str(sys.argv[1])
    DI = str(sys.argv[2])
    noise_region = str(sys.argv[3])
    planloc = str(sys.argv[4])
except IndexError:
    
    tele = "LUVA"
    DI = "ADI"
    noise_region = "dynasquare"
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
    if planet_outside:
        planet_pos_lamD = 21.
        planet_pos_mas = 200.53522829578813
    im_dir_path = "../data/LUVOIR-A_outputs/"
    IWA = 7.5
    OWA = 40#22.
    
if tele == "LUVB":
    planet_pos_lamD = 7.0 # lam/D
    planet_pos_mas = 100.26761414789404
    if planet_outside:
        planet_pos_lamD = 14. # lam/D
        planet_pos_mas = 200.53522829578813
    im_dir_path = "../data/LUVOIR-B_outputs/"
    IWA = 2.5
    OWA = 22 #13

    
    
ap_sz_arr = np.arange(1, 2, 1)
filter_sz_arr_pix = np.arange(1, 102, 1)
im_sz = 101
filter_sz_arr_fourier = im_sz / filter_sz_arr_pix
#filter_sz_arr = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

incl_arr = ["00", "30", "60", "90"]
zodi_arr = ["1", "5", "10", "20", "50", "100"]
longitude = "00"




configs = []

if planet_outside == False:
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


# =============================================================================
# # load height/width file
# hw_fl = "opt_hw_{}_{}".format(tele, DI)
# if planet_outside:
#     hw_fl += "_planout"
# hw_fl+= ".dat"
# hw_df = pd.read_csv(hw_fl, sep=" ", header=0)
# =============================================================================


# define height and width of noise region:
height = 3
width = 2

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
        
# =============================================================================
#     # get height and width:
#     unif_bool = hw_df["uniform_disk"].isin([int(uniform_disk)])
#     ap_sz_bool = hw_df["ap_sz"].isin([float(ap_sz)])
#     filters_ints = hw_df["filter_sz_pix"].values
#     filters_ints = np.rint(filters_ints).astype(int)
#     filter_bool = (filters_ints == int(im_sz/filter_sz))
#     incl_bool =  hw_df["incl"].isin([float(incl)])
#     zodis_bool = hw_df["zodis"].isin([float(zodis)])
#     tot_bool = unif_bool & ap_sz_bool & filter_bool & incl_bool & zodis_bool
#     
#     height = int(hw_df["height"][tot_bool].values[0])
#     width = int(hw_df["width"][tot_bool].values[0])
# =============================================================================
    

    
    
    
    # get planet locations in sci and ref images
    sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, central_pixel, aperture, imsc, diam = ezf.get_planet_locations_and_info(roll_angle, planet_pos_mas, ap_sz, im_dir_path)
    
    # load matched filters according to aperture radius size
    if DI == "ADI":
        matched_filter_fl = matched_filter_dir + "matched_filter_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), ap_sz)
    elif DI == "RDI":
        matched_filter_fl = matched_filter_dir + "matched_filter_single_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), ap_sz)
    matched_filter_datacube = np.load(matched_filter_fl)
    
    
    
    im_dir = im_dir_path + "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}".format(incl, longitude, zodis, round(roll_angle))
    if planet_outside:
        im_dir+= "-planet_outside"
    im_dir += "/"

    cc_SNRs_after_hipass = []
    cc_SNRs_before_hipass = []
    SNR_before_hipass_arr = []
    SNR_after_hipass_arr = []
    
    measured_noise_before_hipass_arr = []
    measured_noise_after_hipass_arr = []
    measured_noise_before_hipass_out_arr = []
    measured_noise_after_hipass_out_arr = []
    
    measured_signal_before_hipass_arr = []
    measured_signal_after_hipass_arr = []
    
    frac_diff_med_arr = []
    frac_diff_std_arr = []
    converged = False
    convergence_counter = 0
    iterations = 0
    
    while not converged:
        iterations += 1
        #print(iterations)
    
        # synthesize images
        if DI == "ADI":
            sci_im, ref_im, sci_planet_counts, ref_planet_counts,  \
            expected_noise_planet, expected_noise_outside, outside_loc = ezf.synthesize_images_ADI(im_dir, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, float(zodis), aperture,
                                                                                                   target_SNR=7, pix_radius=ap_sz,
                                                                                                   verbose=False, 
                                                                                                   add_noise=add_noise, 
                                                                                                   add_star=add_star, 
                                                                                                   planet_noise=planet_noise, 
                                                                                                   uniform_disk=uniform_disk,
                                                                                                   r2_disk=r2_disk)
            sci_out_i, sci_out_j, ref_out_i, ref_out_j = outside_loc
        elif DI == "RDI":
            
            sci_im, ref_im, sci_planet_counts, \
                expected_noise_planet, expected_noise_outside, outside_loc = ezf.synthesize_images_RDI(im_dir, sci_signal_i, sci_signal_j, float(zodis), aperture,
                                                                                                   target_SNR=7, pix_radius=ap_sz, 
                                                                                                   verbose=False,
                                                                                                   add_noise=add_noise, 
                                                                                                   add_star=add_star, 
                                                                                                   planet_noise=planet_noise, 
                                                                                                   uniform_disk=uniform_disk,
                                                                                                   r2_disk=r2_disk)
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
        
       
# =============================================================================
#         ezf.plot_im_ADI(sub_im, sci_signal_i, sci_signal_j, sci_out_i, sci_out_j)
#         assert False
# =============================================================================
                
        # perform high pass filter on the sub im
        sub_im_hipass = ezf.high_pass_filter(sub_im, filtersize=filter_sz)
        
        
        # get expected noise
        expected_noise = np.sqrt(expected_noise_planet)
        expected_noise_out = np.sqrt(expected_noise_outside)
        
        
        # measure noise
        if noise_region == "circle":
            
            if DI == "ADI":
                measured_noise_before_hipass, \
                    nr_circle_sci, nr_circle_sci_opp, \
                    nr_circle_ref, nr_circle_ref_opp = ezf.measure_noise_circle_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                                            sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                                            aperture, ap_sz)
                
                measured_noise_after_hipass, \
                    nr_circle_sci, nr_circle_sci_opp, \
                    nr_circle_ref, nr_circle_ref_opp = ezf.measure_noise_circle_ADI(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                                            sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                                            aperture, ap_sz)
                
                measured_noise_before_hipass_out, \
                    nr_circle_sci_out, nr_circle_sci_opp_out, \
                    nr_circle_ref_out, nr_circle_ref_opp_out = ezf.measure_noise_circle_ADI(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                                    sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                                    aperture, ap_sz)
                
                measured_noise_after_hipass_out, \
                    nr_circle_sci_out, nr_circle_sci_opp_out, \
                    nr_circle_ref_out, nr_circle_ref_opp_out = ezf.measure_noise_circle_ADI(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                                    sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                                    aperture, ap_sz)
            
                noise_map_sci = ~np.isnan(nr_circle_sci) | ~np.isnan(nr_circle_sci_opp)
                noise_map_ref = ~np.isnan(nr_circle_ref) | ~np.isnan(nr_circle_ref_opp)
                
                
            elif DI == "RDI":
                
                measured_noise_before_hipass, \
                    nr_circle_sci, nr_circle_sci_opp = ezf.measure_noise_circle_RDI(sub_im, sci_signal_i, sci_signal_j,
                                                                            sci_signal_i_opp, sci_signal_j_opp, 
                                                                            aperture, ap_sz)
                
                measured_noise_after_hipass, \
                    nr_circle_sci, nr_circle_sci_opp = ezf.measure_noise_circle_RDI(sub_im_hipass, sci_signal_i, sci_signal_j,
                                                                            sci_signal_i_opp, sci_signal_j_opp,
                                                                            aperture, ap_sz)
                
                measured_noise_before_hipass_out, \
                    nr_circle_sci_out, nr_circle_sci_opp_out = ezf.measure_noise_circle_RDI(sub_im, sci_out_i, sci_out_j,
                                                                                    sci_out_i_opp, sci_out_j_opp, 
                                                                                    aperture, ap_sz)
                
                measured_noise_after_hipass_out, \
                    nr_circle_sci_out, nr_circle_sci_opp_out = ezf.measure_noise_circle_RDI(sub_im_hipass, sci_out_i, sci_out_j,
                                                                                    sci_out_i_opp, sci_out_j_opp, 
                                                                                    aperture, ap_sz)
                    
                noise_map_sci = ~np.isnan(nr_circle_sci) | ~np.isnan(nr_circle_sci_opp)
                noise_map_sci_out = ~np.isnan(nr_circle_sci_out) | ~np.isnan(nr_circle_sci_opp_out)
                
            
        elif noise_region == "ring":
            
            measured_noise_before_hipass, nr_ring = ezf.measure_noise_ring(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                              sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                              aperture, ap_sz)
            
            measured_noise_after_hipass, nr_ring = ezf.measure_noise_ring(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                             sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                             aperture, ap_sz)
            
            measured_noise_before_hipass_out, nr_ring_out = ezf.measure_noise_ring(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                  sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                  aperture, ap_sz)
            
            measured_noise_after_hipass_out, nr_ring_out = ezf.measure_noise_ring(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                 sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                 aperture, ap_sz)
            
            noise_map_sci = ~np.isnan(nr_ring)
            noise_map_ref = ~np.isnan(nr_ring)


        
        elif noise_region == "wedge":
            wedge_corrections = False
            
            if DI == "ADI":
                measured_noise_before_hipass, \
                    nr_wedge_sci, nr_wedge_sci_opp, \
                    nr_wedge_ref, nr_wedge_ref_opp = ezf.measure_noise_wedge_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                                         sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                                         aperture, ap_sz, rotation_map, tele, "struct", corrections=wedge_corrections)
                
                measured_noise_after_hipass, \
                    nr_wedge_sci, nr_wedge_sci_opp, \
                    nr_wedge_ref, nr_wedge_ref_opp = ezf.measure_noise_wedge_ADI(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j,
                                                                         sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, 
                                                                         aperture, ap_sz, rotation_map, tele, "struct", corrections=wedge_corrections)
                
                measured_noise_before_hipass_out, \
                    nr_wedge_sci_out, nr_wedge_sci_opp_out, \
                    nr_wedge_ref_out, nr_wedge_ref_opp_out = ezf.measure_noise_wedge_ADI(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                       sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                       aperture, ap_sz, rotation_map, tele, "smooth", corrections=wedge_corrections)
                
                measured_noise_after_hipass_out, \
                    nr_wedge_sci_out, nr_wedge_sci_opp_out, \
                    nr_wedge_ref_out, nr_wedge_ref_opp_out = ezf.measure_noise_wedge_ADI(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j,
                                                                      sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, 
                                                                      aperture, ap_sz, rotation_map, tele, "smooth", corrections=wedge_corrections)
                
                noise_map_sci = ~np.isnan(nr_wedge_sci) | ~np.isnan(nr_wedge_sci_opp)
                noise_map_ref = ~np.isnan(nr_wedge_ref) | ~np.isnan(nr_wedge_ref_opp)
                noise_map_sci_out = ~np.isnan(nr_wedge_sci_out) | ~np.isnan(nr_wedge_sci_opp_out)

                
                
                
                
                
            elif DI == "RDI":
                measured_noise_before_hipass, \
                    nr_wedge_sci, nr_wedge_sci_opp = ezf.measure_noise_wedge_RDI(sub_im, sci_signal_i, sci_signal_j,
                                                                         sci_signal_i_opp, sci_signal_j_opp,
                                                                         aperture, ap_sz, rotation_map, tele, "struct", corrections=wedge_corrections)
                
                measured_noise_after_hipass, \
                    nr_wedge_sci, nr_wedge_sci_opp = ezf.measure_noise_wedge_RDI(sub_im_hipass, sci_signal_i, sci_signal_j,
                                                                         sci_signal_i_opp, sci_signal_j_opp,
                                                                         aperture, ap_sz, rotation_map, tele, "struct", corrections=wedge_corrections)
                
                measured_noise_before_hipass_out, \
                    nr_wedge_sci_out, nr_wedge_sci_opp_out = ezf.measure_noise_wedge_RDI(sub_im, sci_out_i, sci_out_j,
                                                                       sci_out_i_opp, sci_out_j_opp, 
                                                                       aperture, ap_sz, rotation_map, tele, "smooth", corrections=wedge_corrections)
                
                measured_noise_after_hipass_out, \
                    nr_wedge_sci_out, nr_wedge_sci_opp_out = ezf.measure_noise_wedge_RDI(sub_im_hipass, sci_out_i, sci_out_j,
                                                                      sci_out_i_opp, sci_out_j_opp,
                                                                      aperture, ap_sz, rotation_map, tele, "smooth", corrections=wedge_corrections)
                
                noise_map_sci = ~np.isnan(nr_wedge_sci) | ~np.isnan(nr_wedge_sci_opp)
                noise_map_sci_out = ~np.isnan(nr_wedge_sci_out) | ~np.isnan(nr_wedge_sci_opp_out)
                
        elif noise_region == "dynasquare":
            
            
            
                    
            if DI == "ADI":
                measured_noise_before_hipass, \
                    nr_dynasquare_sci, nr_dynasquare_sci_opp, \
                    nr_dynasquare_ref, nr_dynasquare_ref_opp = ezf.measure_noise_dynasquare_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, \
                                                                                 sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, \
                                                                                 aperture, ap_sz, height, width, roll_angle, corrections=False, verbose=False)

                
                measured_noise_after_hipass, \
                    nr_dynasquare_sci, nr_dynasquare_sci_opp, \
                    nr_dynasquare_ref, nr_dynasquare_ref_opp = ezf.measure_noise_dynasquare_ADI(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, \
                                                                                 sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, \
                                                                                 aperture, ap_sz, height, width, roll_angle, corrections=False, verbose=False)

                
                measured_noise_before_hipass_out, \
                    nr_dynasquare_sci_out, nr_dynasquare_sci_opp_out, \
                    nr_dynasquare_ref_out, nr_dynasquare_ref_opp_out = ezf.measure_noise_dynasquare_ADI(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j, \
                                                                                 sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, \
                                                                                 aperture, ap_sz, height, width, roll_angle, corrections=False, verbose=False)

                
                measured_noise_after_hipass_out, \
                    nr_dynasquare_sci_out, nr_dynasquare_sci_opp_out, \
                    nr_dynasquare_ref_out, nr_dynasquare_ref_opp_out = ezf.measure_noise_dynasquare_ADI(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j, \
                                                                                 sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, \
                                                                                 aperture, ap_sz, height, width, roll_angle, corrections=False, verbose=False)

                
                
                noise_map_sci = ~np.isnan(nr_dynasquare_sci) | ~np.isnan(nr_dynasquare_sci_opp)
                noise_map_ref = ~np.isnan(nr_dynasquare_ref) | ~np.isnan(nr_dynasquare_ref_opp)
                noise_map_sci_out = ~np.isnan(nr_dynasquare_sci_out) | ~np.isnan(nr_dynasquare_sci_opp_out)
                
            elif DI == "RDI":
                measured_noise_before_hipass, \
                    nr_dynasquare_sci, nr_dynasquare_sci_opp = ezf.measure_noise_dynasquare_RDI(sub_im, sci_signal_i, sci_signal_j,
                                                                                 sci_signal_i_opp, sci_signal_j_opp, 
                                                                                 aperture, ap_sz, height, width, corrections=False, verbose=False)

                
                measured_noise_after_hipass, \
                    nr_dynasquare_sci, nr_dynasquare_sci_opp = ezf.measure_noise_dynasquare_RDI(sub_im_hipass, sci_signal_i, sci_signal_j,
                                                                                 sci_signal_i_opp, sci_signal_j_opp,
                                                                                 aperture, ap_sz, height, width, corrections=False, verbose=False)

                
                measured_noise_before_hipass_out, \
                    nr_dynasquare_sci_out, nr_dynasquare_sci_opp_out = ezf.measure_noise_dynasquare_RDI(sub_im, sci_out_i, sci_out_j, 
                                                                                 sci_out_i_opp, sci_out_j_opp, 
                                                                                 aperture, ap_sz, height, width, corrections=False, verbose=False)

                
                measured_noise_after_hipass_out, \
                    nr_dynasquare_sci_out, nr_dynasquare_sci_opp_out = ezf.measure_noise_dynasquare_RDI(sub_im_hipass, sci_out_i, sci_out_j,
                                                                                 sci_out_i_opp, sci_out_j_opp, 
                                                                                 aperture, ap_sz, height, width, corrections=False, verbose=False)
               
                
                
                noise_map_sci = ~np.isnan(nr_dynasquare_sci) | ~np.isnan(nr_dynasquare_sci_opp)
                noise_map_sci_out = ~np.isnan(nr_dynasquare_sci_out) | ~np.isnan(nr_dynasquare_sci_opp_out)
            
                    
                    
            
        
# =============================================================================
#         ezf.plot_im_ADI(sub_im_hipass*noise_map_sci_out, sci_signal_i, sci_signal_j, sci_out_i, sci_out_j)
#         assert False
# =============================================================================
        
        # measure signal
        if DI == "ADI":
            signal_before_hipass = ezf.measure_signal_ADI(sub_im, ~np.isnan(nr_dynasquare_sci), ~np.isnan(nr_dynasquare_ref), sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture)
            signal_after_hipass = ezf.measure_signal_ADI(sub_im_hipass, ~np.isnan(nr_dynasquare_sci), ~np.isnan(nr_dynasquare_ref), sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture)

        elif DI == "RDI":
            signal_before_hipass = ezf.measure_signal_RDI(sub_im, ~np.isnan(nr_dynasquare_sci), sci_signal_i, sci_signal_j, aperture)
            signal_after_hipass = ezf.measure_signal_RDI(sub_im_hipass, ~np.isnan(nr_dynasquare_sci), sci_signal_i, sci_signal_j, aperture)
            
        
        
        measured_signal_before_hipass_arr.append(signal_before_hipass)
        measured_signal_after_hipass_arr.append(signal_after_hipass)

        SNR_before_hipass = signal_before_hipass / measured_noise_before_hipass
        SNR_after_hipass = signal_after_hipass / measured_noise_after_hipass
        
        SNR_before_hipass_arr.append(SNR_before_hipass)
        SNR_after_hipass_arr.append(SNR_after_hipass)
        
        #print("SNR before hipass:", SNR_before_hipass)
        #print("SNR after hipass:", SNR_after_hipass)

            
        
        measured_noise_before_hipass_arr.append(measured_noise_before_hipass)
        measured_noise_after_hipass_arr.append(measured_noise_after_hipass)
        measured_noise_before_hipass_out_arr.append(measured_noise_before_hipass_out)
        measured_noise_after_hipass_out_arr.append(measured_noise_after_hipass_out)
        
        
        

        

        # cross correlation
        
        cc_map_before_hipass = ezf.calculate_cc_map(matched_filter_datacube, sub_im, valid_mask)
        cc_map_after_hipass = ezf.calculate_cc_map(matched_filter_datacube, sub_im_hipass, valid_mask)
                
        

        if DI == "ADI":
            cc_SNR_before_hipass = ezf.calc_CC_SNR_ADI(cc_map_before_hipass, noise_map_sci, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, ref_signal_i_opp, ref_signal_j_opp, ap_sz, noise_region)
            cc_SNR_after_hipass = ezf.calc_CC_SNR_ADI(cc_map_after_hipass, noise_map_sci, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, ref_signal_i_opp, ref_signal_j_opp, ap_sz, noise_region)
        elif DI == "RDI":
            cc_SNR_before_hipass = ezf.calc_CC_SNR_RDI(cc_map_before_hipass, noise_map_sci, sci_signal_i, sci_signal_j, ap_sz, noise_region)
            cc_SNR_after_hipass = ezf.calc_CC_SNR_RDI(cc_map_after_hipass, noise_map_sci, sci_signal_i, sci_signal_j, ap_sz, noise_region)
            
        
        cc_SNRs_before_hipass.append(cc_SNR_before_hipass)
                
        cc_SNRs_after_hipass.append(cc_SNR_after_hipass)
        

        if iterations == 500:
            print("NOT CONVERGED: Iteration limit reached.")
            break
        
    median_SNR_before_hipass = np.median(SNR_before_hipass_arr)
    median_SNR_after_hipass = np.median(SNR_after_hipass_arr)
    median_cc_SNR_after_hipass = np.median(cc_SNRs_after_hipass)
    median_cc_SNR_before_hipass = np.median(cc_SNRs_before_hipass)
    median_measured_noise_before_hipass = np.median(measured_noise_before_hipass_arr)
    median_measured_noise_after_hipass = np.median(measured_noise_after_hipass_arr)
    median_measured_noise_before_hipass_out = np.median(measured_noise_before_hipass_out_arr)
    median_measured_noise_after_hipass_out = np.median(measured_noise_after_hipass_out_arr)
    median_measured_signal_before_hipass = np.median(measured_signal_before_hipass_arr)
    median_measured_signal_after_hipass = np.median(measured_signal_after_hipass_arr)

    std_SNR_after_hipass = np.std(SNR_after_hipass_arr)
    std_cc_SNR_after_hipass = np.std(cc_SNRs_after_hipass)
    std_noise_after_hipass = np.std(measured_noise_after_hipass_arr)
    std_noise_after_hipass_out = np.std(measured_noise_after_hipass_out_arr)
    std_signal_after_hipass = np.std(measured_signal_after_hipass_arr)

    
    verbose = False
    if verbose:
        print("Median SNR before hipass:", median_SNR_before_hipass)
        print("Median SNR after hipass:", median_SNR_after_hipass)
        print("Median CC SNR before hipass:", median_cc_SNR_before_hipass)
        print("Median CC SNR after hipass:", median_cc_SNR_after_hipass)
        print("Expected noise:", expected_noise)
        print("Expected noise out:", expected_noise_out)
        print("Median measured/expected noise before hipass:", median_measured_noise_before_hipass/expected_noise)
        print("Median measured/expected noise after hipass:", median_measured_noise_after_hipass/expected_noise)
        print("Median measured/expected noise before hipass outside:", median_measured_noise_before_hipass_out/expected_noise_out)
        print("Median measured/expected noise after hipass outside:", median_measured_noise_after_hipass_out/expected_noise_out)
        #assert False

    return_arr = np.array([uniform_disk, ap_sz, im_sz/filter_sz, int(incl), int(zodis), 
                           median_SNR_before_hipass, median_SNR_after_hipass,
                           median_cc_SNR_after_hipass, median_cc_SNR_before_hipass, iterations,
                           median_measured_noise_before_hipass, median_measured_noise_after_hipass, expected_noise,
                           median_measured_noise_before_hipass_out, median_measured_noise_after_hipass_out, expected_noise_out,
                           median_measured_signal_before_hipass, median_measured_signal_after_hipass, 
                           std_SNR_after_hipass, std_cc_SNR_after_hipass, std_noise_after_hipass, 
                           std_noise_after_hipass_out, std_signal_after_hipass])
    
    end_time = time.time()
    #print("Time elapsed for process: {} s".format(round(end_time - start_time, 2)))
    return return_arr


parallel = False

# sequential runs
if parallel == False:
    data = []
    
    #configs = [([1, 10, "00", "1", "uniform"])]
    configs = [([1, 101/101., "30", "20", "model"])]
    for config in configs:
        
        data_arr  = process(config)
        data.append(data_arr)
        print(data_arr)
        
    data = np.array(data)
    
# =============================================================================
#     header = "ap_sz filter_sz incl zodis median_cc_SNR median_cc_SNR_before_hipass iterations measured_noise_before_hipass measured_noise_after_hipass expected_noise"
#     np.savetxt("data.dat", data, header=header, comments='')
# =============================================================================
    
        

# parallel runs
elif parallel == True:
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=39)(delayed(process)(config) for config in configs)
    
    header = "uniform_disk ap_sz filter_sz_pix incl zodis median_SNR_before_hipass median_SNR_after_hipass median_cc_SNR_after_hipass median_cc_SNR_before_hipass iterations measured_noise_before_hipass measured_noise_after_hipass expected_noise median_measured_noise_before_hipass_out median_measured_noise_after_hipass_out expected_noise_out median_measured_signal_before_hipass median_measured_signal_after_hipass std_SNR_after_hipass std_cc_SNR_after_hipass std_noise_after_hipass std_noise_after_hipass_out std_signal_after_hipass"
    save_fl = "data_{}_{}_{}".format(tele, DI, noise_region)
    if planet_outside:
        save_fl += "_planout"
    save_fl += ".dat"
    np.savetxt(save_fl, results, header=header, comments='')



                
                
