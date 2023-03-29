# -*- coding: utf-8 -*-

import exozodi_functions as ezf
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import sys


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
except IndexError:
    
    tele = "LUVB"
    DI = "ADI"
    noise_region = "dynasquare"
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
    IWA = 2.5
    OWA = 22 #13

    
    
ap_sz_arr = np.arange(1, 3, 1)
filter_sz_arr_pix = np.arange(1, 102, 1)
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

import time
def process(config):
    start_time = time.time()
    
    ap_sz, filter_sz, incl, zodis, disk_type = config
    print(disk_type, ap_sz, filter_sz, incl, zodis)
    
    # get planet locations in sci and ref images
    sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, central_pixel, aperture, imsc, diam = ezf.get_planet_locations_and_info(roll_angle, planet_pos_mas, ap_sz, im_dir_path)
    
    # load matched filters according to aperture radius size
    if DI == "ADI":
        matched_filter_fl = matched_filter_dir + "matched_filter_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), ap_sz)
    elif DI == "RDI":
        matched_filter_fl = matched_filter_dir + "matched_filter_single_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), ap_sz)
    matched_filter_datacube = np.load(matched_filter_fl)
    
    if disk_type == "uniform":
        uniform_disk = True
    elif disk_type == "model":
        uniform_disk = False
    else:
        assert False, "disk type not recognized"
    
    im_dir = im_dir_path + "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}/".format(incl, longitude, zodis, round(roll_angle))
    

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
            
            hwn_before_hipass = []
            hwn_after_hipass = []
            hwn_before_hipass_out = []
            hwn_after_hipass_out = []
            
            for height in np.arange(ap_sz*2, 2*(ap_sz*2+1)+1, 1, dtype=int):
                for width in np.arange(ap_sz*2, 2*(ap_sz*2+1)+1, 1, dtype=int):
                    #print(c, height, width)
                    
                    if DI == "ADI":
                        measured_noise_before_hipass, \
                            nr_dynasquare_sci, nr_dynasquare_sci_opp, \
                            nr_dynasquare_ref, nr_dynasquare_ref_opp = ezf.measure_noise_dynasquare_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, \
                                                                                         sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, \
                                                                                         aperture, ap_sz, width, height, corrections=False, verbose=False)
        
                        
                        measured_noise_after_hipass, \
                            nr_dynasquare_sci, nr_dynasquare_sci_opp, \
                            nr_dynasquare_ref, nr_dynasquare_ref_opp = ezf.measure_noise_dynasquare_ADI(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, \
                                                                                         sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, \
                                                                                         aperture, ap_sz, width, height, corrections=False, verbose=False)
        
                        
                        measured_noise_before_hipass_out, \
                            nr_dynasquare_sci_out, nr_dynasquare_sci_opp_out, \
                            nr_dynasquare_ref_out, nr_dynasquare_ref_opp_out = ezf.measure_noise_dynasquare_ADI(sub_im, sci_out_i, sci_out_j, ref_out_i, ref_out_j, \
                                                                                         sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, \
                                                                                         aperture, ap_sz, width, height, corrections=False, verbose=False)
        
                        
                        measured_noise_after_hipass_out, \
                            nr_dynasquare_sci_out, nr_dynasquare_sci_opp_out, \
                            nr_dynasquare_ref_out, nr_dynasquare_ref_opp_out = ezf.measure_noise_dynasquare_ADI(sub_im_hipass, sci_out_i, sci_out_j, ref_out_i, ref_out_j, \
                                                                                         sci_out_i_opp, sci_out_j_opp, ref_out_i_opp, ref_out_j_opp, \
                                                                                         aperture, ap_sz, width, height, corrections=False, verbose=False)
        
                        hwn_before_hipass.append([height, width, measured_noise_before_hipass])
                        hwn_after_hipass.append([height, width, measured_noise_after_hipass])
                        hwn_before_hipass_out.append([height, width, measured_noise_before_hipass_out])
                        hwn_after_hipass_out.append([height, width, measured_noise_after_hipass_out])
                        
                        noise_map_sci = ~np.isnan(nr_dynasquare_sci) | ~np.isnan(nr_dynasquare_sci_opp)
                        noise_map_ref = ~np.isnan(nr_dynasquare_ref) | ~np.isnan(nr_dynasquare_ref_opp)
                        noise_map_sci_out = ~np.isnan(nr_dynasquare_sci_out) | ~np.isnan(nr_dynasquare_sci_opp_out)
                        
                    elif DI == "RDI":
                        measured_noise_before_hipass, \
                            nr_dynasquare_sci, nr_dynasquare_sci_opp = ezf.measure_noise_dynasquare_RDI(sub_im, sci_signal_i, sci_signal_j,
                                                                                         sci_signal_i_opp, sci_signal_j_opp, 
                                                                                         aperture, ap_sz, width, height, corrections=False, verbose=False)
        
                        
                        measured_noise_after_hipass, \
                            nr_dynasquare_sci, nr_dynasquare_sci_opp = ezf.measure_noise_dynasquare_RDI(sub_im_hipass, sci_signal_i, sci_signal_j,
                                                                                         sci_signal_i_opp, sci_signal_j_opp,
                                                                                         aperture, ap_sz, width, height, corrections=False, verbose=False)
        
                        
                        measured_noise_before_hipass_out, \
                            nr_dynasquare_sci_out, nr_dynasquare_sci_opp_out = ezf.measure_noise_dynasquare_RDI(sub_im, sci_out_i, sci_out_j, 
                                                                                         sci_out_i_opp, sci_out_j_opp, 
                                                                                         aperture, ap_sz, width, height, corrections=False, verbose=False)
        
                        
                        measured_noise_after_hipass_out, \
                            nr_dynasquare_sci_out, nr_dynasquare_sci_opp_out = ezf.measure_noise_dynasquare_RDI(sub_im_hipass, sci_out_i, sci_out_j,
                                                                                         sci_out_i_opp, sci_out_j_opp, 
                                                                                         aperture, ap_sz, width, height, corrections=False, verbose=False)
                       
                        hwn_before_hipass.append([height, width, measured_noise_before_hipass])
                        hwn_after_hipass.append([height, width, measured_noise_after_hipass])
                        hwn_before_hipass_out.append([height, width, measured_noise_before_hipass_out])
                        hwn_after_hipass_out.append([height, width, measured_noise_after_hipass_out])
                        
                        noise_map_sci = ~np.isnan(nr_dynasquare_sci) | ~np.isnan(nr_dynasquare_sci_opp)
                        noise_map_sci_out = ~np.isnan(nr_dynasquare_sci_out) | ~np.isnan(nr_dynasquare_sci_opp_out)
                    
                    
                    
                    
            hwn_before_hipass = np.array(hwn_before_hipass)
            hwn_after_hipass = np.array(hwn_after_hipass)
            hwn_before_hipass_out = np.array(hwn_before_hipass_out)
            hwn_after_hipass_out = np.array(hwn_after_hipass_out)
            
            opt_n_before_hipass_ind = np.argmin(np.abs(hwn_before_hipass.T[2]/expected_noise-1))
            opt_n_after_hipass_ind = np.argmin(np.abs(hwn_after_hipass.T[2]/expected_noise-1))
            opt_n_before_hipass_out_ind = np.argmin(np.abs(hwn_before_hipass_out.T[2]/expected_noise_out-1))
            opt_n_after_hipass_out_ind = np.argmin(np.abs(hwn_after_hipass_out.T[2]/expected_noise_out-1))
            
            h_before_hipass, w_before_hipass, measured_noise_before_hipass = hwn_before_hipass[opt_n_before_hipass_ind]
            h_after_hipass, w_after_hipass, measured_noise_after_hipass = hwn_after_hipass[opt_n_after_hipass_ind]
            h_before_hipass_out, w_before_hipass_out, measured_noise_before_hipass_out = hwn_before_hipass_out[opt_n_before_hipass_out_ind]
            h_after_hipass_out, w_after_hipass_out, measured_noise_after_hipass_out = hwn_after_hipass_out[opt_n_after_hipass_out_ind]
            
        
# =============================================================================
#         ezf.plot_im_ADI(sub_im_hipass*noise_map_sci_out, sci_signal_i, sci_signal_j, sci_out_i, sci_out_j)
#         assert False
# =============================================================================
        
        # measure signal
        if DI == "ADI":
            signal_before_hipass = ezf.measure_signal_ADI(sub_im, noise_map_sci, noise_map_ref, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture)
            signal_after_hipass = ezf.measure_signal_ADI(sub_im_hipass, noise_map_sci, noise_map_ref, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, aperture)
        elif DI == "RDI":
            signal_before_hipass = ezf.measure_signal_RDI(sub_im, noise_map_sci, sci_signal_i, sci_signal_j, aperture)
            signal_after_hipass = ezf.measure_signal_RDI(sub_im_hipass, noise_map_sci, sci_signal_i, sci_signal_j, aperture)
            
        
        
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
        
# =============================================================================
#         if iterations > 1:
#             # compute fractional difference of medians
#             old_median = np.median(cc_SNRs_after_hipass[:-1])
#             new_median = np.median(cc_SNRs_after_hipass)
#             
#             frac_diff_med = np.abs((new_median - old_median) / old_median)
#             #print("med:",old_median, new_median, frac_diff_med)
#             
#             old_std = np.std(cc_SNRs_after_hipass[:-1])
#             new_std = np.std(cc_SNRs_after_hipass)
#             frac_diff_std = np.abs((new_std - old_std) / old_std)
#             #print("std:", old_std, new_std, frac_diff_std)
#             
#             frac_diff_med_arr.append(frac_diff_med)
#             frac_diff_std_arr.append(frac_diff_std)
# 
#         if iterations > 500:
#             
# 
#             if (frac_diff_med < 0.01) and (frac_diff_std < 0.01):
#                 #print("Converged")
#                 convergence_counter += 1
#                 if convergence_counter == 10:
#                     print("Converged")
#                     converged = True
#                     
#             else:
#                 # reset the convergence counter
#                 convergence_counter = 0
# =============================================================================

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

    print(median_measured_noise_after_hipass_out / expected_noise_out)
    return_arr = np.array([uniform_disk, ap_sz, im_sz/filter_sz, int(incl), int(zodis), 
                           median_SNR_before_hipass, median_SNR_after_hipass,
                           median_cc_SNR_after_hipass, median_cc_SNR_before_hipass, iterations,
                           median_measured_noise_before_hipass, median_measured_noise_after_hipass, expected_noise,
                           median_measured_noise_before_hipass_out, median_measured_noise_after_hipass_out, expected_noise_out,
                           median_measured_signal_before_hipass, median_measured_signal_after_hipass])
    
    end_time = time.time()
    print("Time elapsed for process: {} s".format(round(end_time - start_time, 2)))
    return return_arr


parallel = True

# sequential runs
if parallel == False:
    data = []
    
    configs = [([1, 10, "00", "1", "uniform"])]
    configs = [([1, 101., "60", "100", "model"])]
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
    
    header = "uniform_disk ap_sz filter_sz_pix incl zodis median_SNR_before_hipass median_SNR_after_hipass median_cc_SNR_after_hipass median_cc_SNR_before_hipass iterations measured_noise_before_hipass measured_noise_after_hipass expected_noise median_measured_noise_before_hipass_out median_measured_noise_after_hipass_out expected_noise_out median_measured_signal_before_hipass median_measured_signal_after_hipass"
    np.savetxt("data_{}_{}_{}_test.dat".format(tele, DI, noise_region), results, header=header, comments='')



                
                
