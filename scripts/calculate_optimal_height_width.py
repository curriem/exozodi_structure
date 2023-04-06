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
planet_outside = True

try:
    tele = str(sys.argv[1])
    DI = str(sys.argv[2])
except IndexError:
    
    tele = "LUVA"
    DI = "ADI"
    print("WARNING: NO TELE, DI, NOISE REGION SPECIFIED. USING {}, {}.".format(tele, DI))

matched_filter_dir = "../matched_filter_library/"


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
    
    heights = []
    widths = []

    
    if disk_type == "uniform":
        uniform_disk = True
    elif disk_type == "model":
        uniform_disk = False
    else:
        assert False, "disk type not recognized"
    
    im_dir = im_dir_path + "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}".format(incl, longitude, zodis, round(roll_angle))
    if planet_outside:
        im_dir+= "-planet_outside"
    im_dir += "/"

    
    converged = False
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
            
        
        hwn_after_hipass = []
 
        
        for height in np.arange(ap_sz*2, 2*(ap_sz*2+1)+1, 1, dtype=int):
            for width in np.arange(ap_sz*2, 2*(ap_sz*2+1)+1, 1, dtype=int):
                
                if DI == "ADI":

                    measured_noise_after_hipass, \
                        nr_dynasquare_sci, nr_dynasquare_sci_opp, \
                        nr_dynasquare_ref, nr_dynasquare_ref_opp = ezf.measure_noise_dynasquare_ADI(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, \
                                                                                     sci_signal_i_opp, sci_signal_j_opp, ref_signal_i_opp, ref_signal_j_opp, \
                                                                                     aperture, ap_sz, width, height, corrections=False, verbose=False)
    
                    

    
                    hwn_after_hipass.append([height, width, measured_noise_after_hipass])
                    
                    
                    
                elif DI == "RDI":
                    
                    
                    measured_noise_after_hipass, \
                        nr_dynasquare_sci, nr_dynasquare_sci_opp = ezf.measure_noise_dynasquare_RDI(sub_im_hipass, sci_signal_i, sci_signal_j,
                                                                                     sci_signal_i_opp, sci_signal_j_opp,
                                                                                     aperture, ap_sz, width, height, corrections=False, verbose=False)
    
                    hwn_after_hipass.append([height, width, measured_noise_after_hipass])
                    
                    
                
                
        
        hwn_after_hipass = np.array(hwn_after_hipass)
        
        
        
        opt_n_after_hipass_ind = np.argmin(np.abs(hwn_after_hipass.T[2]/expected_noise-1))
        
        
        
        h_after_hipass, w_after_hipass, measured_noise_after_hipass = hwn_after_hipass[opt_n_after_hipass_ind]
        

        heights.append(h_after_hipass)
        widths.append(w_after_hipass)
        
        if iterations == 1000:
            print("NOT CONVERGED: Iteration limit reached.")
            print("Time elapsed:", round(time.time() - start_time, 2))
            break
        
    fig, axes = plt.subplots(2, 1)
    axes[0].hist(heights)
    axes[1].hist(widths)
    plt.show()
    
    
    return_arr = np.array([uniform_disk, ap_sz, im_sz/filter_sz, int(incl), int(zodis), 
                           np.median(heights), np.median(widths)])
    
    return return_arr
    
parallel = True

# sequential runs
if parallel == False:
    data = []
    
    #configs = [([1, 10, "00", "1", "uniform"])]
    configs = [([1, 101/1., "00", "1", "model"])]
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
    
    results = Parallel(n_jobs=39)(delayed(process)(config) for config in configs[:5])
    
    header = "uniform_disk ap_sz filter_sz_pix incl zodis height width"
    save_fl = "opt_hw_{}_{}".format(tele, DI)
    if planet_outside:
        save_fl += "_planout"
    save_fl += ".dat"
    np.savetxt(save_fl, results, header=header, comments='')


    