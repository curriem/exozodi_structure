#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import exozodi_functions as ezf
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
"""
Created on Fri Jan 13 15:17:35 2023

@author: mcurr
"""

# choosing noise region

# define some parameters
tele = "LUVA" # telescope


zodis = "10" # zodi level you want to work with
incl = "00"
longitude = "00"
pix_radius = 5
roll_angle = 90.

zodis_arr = ["1", "5", "10", "20", "50", "100"]
incl_arr = ["00", "30", "60", "90"]

matched_filter_dir = "/Users/mcurr/PROJECTS/exozodi_structure/matched_filter_library/"

matched_filter_fl = matched_filter_dir + "matched_filter_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), pix_radius)
matched_filter_datacube = np.load(matched_filter_fl)

matched_filter_single_fl = matched_filter_dir + "matched_filter_single_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), pix_radius)
matched_filter_single_datacube = np.load(matched_filter_single_fl)




planet_pos_lamD = 10.5
planet_pos_mas = 100.26761414789404
    

#im_dir = "/Users/mcurr/PACKAGES/coroSims/LUVOIR-A_outputs/"
im_dir = "/Users/mcurr/PROJECTS/exozodi_structure/data/LUVOIR-A_outputs/"
im_dir += "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}/".format(incl, longitude, zodis, round(roll_angle))

# open an image just to get some information about it
sci_im_fits = pyfits.open(im_dir + "/DET/sci_imgs.fits")
sci_im = sci_im_fits[0].data[0, 0]
imsc = sci_im_fits[0].header["IMSC"] # lam/D
imsz = sci_im_fits[0].header["IMSZ"] # pix
wave = sci_im_fits["WAVE"].data[0]
diam = sci_im_fits[0].header["DIAM"]


central_pixel = (imsz - 1)/2
loc_of_planet_pix = planet_pos_mas/imsc
print(central_pixel+loc_of_planet_pix)
sci_x = loc_of_planet_pix
sci_y = 0 

sci_signal_i = round(central_pixel)
sci_signal_j = round(central_pixel + loc_of_planet_pix)
print(central_pixel, sci_signal_j)


aperture = ezf.get_psf_stamp(np.copy(sci_im), sci_signal_i, sci_signal_j, pix_radius) > 0


ref_x = loc_of_planet_pix * np.sin(np.deg2rad(roll_angle))
ref_y = loc_of_planet_pix * np.cos(np.deg2rad(roll_angle))

ref_signal_i = round(ref_x + central_pixel)
ref_signal_j = round(ref_y + central_pixel)





###############################################################################
add_noise = True
add_star = True
planet_noise = True
uniform_disk = False
r2_disk = False
plot = False
plot_median = True
verbose = False
r2_correct = False
# set noise seed
#np.random.seed(0)

#### r2_correct = True for everything except uniform disk

##### PLOTTING ######
iteration_plot = True

def plot_SNR_vs_iterations(iterations_arr, frac_diff_med_arr, frac_diff_std_arr):
    
    
    plt.figure()
    plt.plot(iterations_arr, frac_diff_med_arr, label="med")
    plt.plot(iterations_arr, frac_diff_std_arr, label="std")

    plt.axhline(0.01, color="k", linestyle=":")
    plt.xlabel("\# Iterations")
    plt.ylabel(r"|$\Delta$SNR/SNR|")
    plt.legend()
    plt.show()
    
    




sub_SNRs = []
sub_SNRs_hipass = []
cc_SNRs = []
sci_SNRs = []
ref_SNRs = []
cc_sci_SNRs = []

sci_ims = []
ref_ims = []
sub_ims = []
sub_hipass_ims = []
cc_sci_maps = []
cc_maps = []

cc_median_vals = []
frac_diff_med_arr = []
frac_diff_std_arr = []


optimal_filtersize_unknown = True


iteration_arr = []

converged = False
convergence_counter = 0

iterations = 0

outer_rs = np.arange(5, 21, 1)




fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
incls = ["00", "30", "60", "90"]
zodi_levels = ["1", "5", "10", "20", "50", "100"]

for a, axis in enumerate(axes.flatten()):
    axis.set_title("inclination {}".format(incls[a]))
    axis.set_yscale("log")
    
axes[0,0].set_ylabel("measured/expected noise")
axes[1,0].set_ylabel("measured/expected noise")

axes[1,0].set_xlabel("Noise region radius (\# pix)")
axes[1,1].set_xlabel("Noise region radius (\# pix)")


for i_incl, incl in enumerate(incls):
    for i_zodis, zodis in enumerate(zodi_levels):
        im_dir = "/Users/mcurr/PROJECTS/exozodi_structure/data/LUVOIR-A_outputs/"
        im_dir += "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10-rang_{}/".format(incl, longitude, zodis, round(roll_angle))

        noise_vals_region_radius = []
        SNR_vals_region_radius = []
        signal_vals_region_radius = []

        for iterations in range(10):
            
        
        
        #for n in range(num_iter):
            print(iterations, "incl", incl, "zodis", zodis)
            sci_im, ref_im, sci_planet_counts, ref_planet_counts, tot_noise_counts = ezf.synthesize_images(im_dir, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, float(zodis), aperture,
                                               target_SNR=7, pix_radius=pix_radius,
                                               verbose=False, 
                                               add_noise=add_noise, 
                                               add_star=add_star, 
                                               planet_noise=planet_noise, 
                                               uniform_disk=uniform_disk,
                                               r2_disk=r2_disk)
            print(tot_noise_counts)
            
            total_planet_counts = sci_planet_counts + ref_planet_counts
            
            # calculate maps
            rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im, imsc, diam, IWA_lamD=7.5, plotting=False)
        
        
        
            sci_im[~valid_mask] = np.nan
            ref_im[~valid_mask] = np.nan
        
        
            im = sci_im
            signal_i = sci_signal_i
            signal_j = sci_signal_j
            valid_map = valid_mask
            # CALCULATE SNR PLAN AN $$$$$$$
            imsz, imsz = im.shape
            apsz, apsz = aperture.shape
            ap_rad = int((apsz - 1)/2)
            
            signal_mask = np.zeros_like(im, dtype=bool)
            signal_mask[signal_i-ap_rad:signal_i+ap_rad+1, signal_j-ap_rad: signal_j+ap_rad+1] = aperture
            
            
            valid_map_signal = np.ones_like(im, dtype=bool)
            valid_map_signal[signal_i-2*ap_rad:signal_i+2*ap_rad+1, signal_j-2*ap_rad: signal_j+2*ap_rad+1] = False
            
            valid_map = valid_map & valid_map_signal
        
            
            noises = []
            SNRs = []
            signals = []
            for outer_r in outer_rs:
                noise_region = ezf.calculate_noise_region_plan_an(im, signal_i, signal_j, inner_r=ap_rad, outer_r=outer_r)
# =============================================================================
#                 if outer_r == 10:
#                     plt.figure()
#                     plt.imshow(noise_region)
# =============================================================================
                noise_region_median = np.nanmedian(noise_region)
                
                noise_region_bkgr_rm = noise_region - noise_region_median
                
                noise = ezf.calc_noise_in_region_two_apertures(noise_region_bkgr_rm, aperture, ap_rad)
                noises.append(noise)
                im_bkgr_sub = im - noise_region_median
                
                signal = np.sum(im_bkgr_sub[signal_mask])
                signals.append(signal)
                
                SNR = signal / noise
                SNRs.append(SNR)
        
            ###############################
            noise_vals_region_radius.append(np.array(noises)/np.sqrt(tot_noise_counts))
            SNR_vals_region_radius.append(SNRs)
            signal_vals_region_radius.append(signals)
            
        
        noise_vals_region_radius = np.array(noise_vals_region_radius)
        SNR_vals_region_radius = np.array(SNR_vals_region_radius)
        signal_vals_region_radius = np.array(signal_vals_region_radius)
        
        
        noise_vals_region_radius_med = np.median(noise_vals_region_radius, axis=0)
        SNR_vals_region_radius_med = np.median(SNR_vals_region_radius, axis=0)
        signal_vals_region_radius_med = np.median(signal_vals_region_radius, axis=0)
        
        axes.flatten()[i_incl].plot(outer_rs, noise_vals_region_radius_med, label=zodi_levels[i_zodis])


axes.flatten()[-1].legend(title="zodis", loc="lower right")
plt.tight_layout()
plt.savefig("../plots/noise_region_inner{}.png".format(pix_radius))
assert False

plt.figure()
plt.plot(outer_rs, noise_vals_region_radius_med)
plt.xlabel("outer radius (num pix)")
plt.ylabel("noise val")

plt.figure()
plt.plot(outer_rs, SNR_vals_region_radius_med)
plt.xlabel("outer radius (num pix)")
plt.ylabel("SNR val")

plt.figure()
plt.plot(outer_rs, signal_vals_region_radius_med)
plt.xlabel("outer radius (num pix)")
plt.ylabel("signal val")

plt.show()
assert False

        

if iteration_plot:
    iterations_arr = np.arange(2, len(cc_SNRs)+1, 1)
    plot_SNR_vs_iterations(iterations_arr, frac_diff_med_arr, frac_diff_std_arr)

def get_closest_ind_to_median(arr):
    
    median_val = np.median(arr)
    difference_array = np.absolute(arr-median_val)
    closest_ind = difference_array.argmin()
    
    return closest_ind

def plot_median_maps():

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    
    sci_SNR_closest_ind = get_closest_ind_to_median(sci_SNRs)
    
    sci_im_plot = axes[0, 0].imshow(np.log10(sci_ims[sci_SNR_closest_ind]), origin='lower')
    #sci_im_plot = axes[0, 0].imshow(sci_ims[sci_SNR_closest_ind], origin='lower')

    axes[0, 0].set_title("Science Image")
    plt.colorbar(sci_im_plot, ax=axes[0,0])
    axes[0, 0].text(2, 95, "SNR={}".format(round(np.median(sci_SNRs), 2)))
    
    
    
    ref_SNR_closest_ind = get_closest_ind_to_median(ref_SNRs)

    #ref_im_plot = axes[1, 0].imshow(ref_ims[ref_SNR_closest_ind], origin='lower')
    ref_im_plot = axes[1, 0].imshow(np.log10(ref_ims[ref_SNR_closest_ind]), origin='lower')
    axes[1, 0].set_title("Reference Image")
    plt.colorbar(ref_im_plot, ax=axes[1,0])
    axes[1, 0].text(2, 95, "SNR={}".format(round(np.median(ref_SNRs), 2)))



    sub_SNR_closest_ind = get_closest_ind_to_median(sub_SNRs)
    
    #sub_im_plot = axes[0, 1].imshow(np.log10(sub_ims[sub_SNR_closest_ind]), origin='lower')
    sub_im_plot = axes[0, 1].imshow(sub_ims[sub_SNR_closest_ind], origin='lower')
    axes[0, 1].set_title("Roll Subtracted Image")
    plt.colorbar(sub_im_plot, ax=axes[0, 1])
    axes[0, 1].text(2, 95, "SNR={}".format(round(np.median(sub_SNRs), 2)))

    

    sub_hipass_SNR_closest_ind = get_closest_ind_to_median(sub_SNRs_hipass)

    #sub_im_hipass_plot = axes[1, 1].imshow(np.log10(sub_hipass_ims[sub_hipass_SNR_closest_ind]), origin='lower')
    sub_im_hipass_plot = axes[1, 1].imshow(sub_hipass_ims[sub_hipass_SNR_closest_ind], origin='lower')
    axes[1, 1].set_title("Hipass Roll Sub Image")
    plt.colorbar(sub_im_hipass_plot, ax=axes[1,1])
    axes[1, 1].text(2, 95, "SNR={}".format(round(np.median(sub_SNRs_hipass), 2)))
    
    
    
    cc_SNR_closest_ind = get_closest_ind_to_median(cc_SNRs)
    
    #cc_map_plot = axes[1, 2].imshow(np.log10(cc_maps[cc_SNR_closest_ind]), origin='lower')
    cc_map_plot = axes[1, 2].imshow(cc_maps[cc_SNR_closest_ind], origin='lower')

    axes[1, 2].set_title("Cross-correlation Map")
    plt.colorbar(cc_map_plot, ax=axes[1,2])
    axes[1, 2].text(2, 95, "SNR={}".format(round(np.median(cc_SNRs), 2)))
    
    
    cc_sci_SNR_closest_ind = get_closest_ind_to_median(cc_sci_maps)
    
    #cc_map_sci_plot = axes[0, 2].imshow(np.log10(cc_sci_maps[cc_sci_SNR_closest_ind]), origin='lower')
    cc_map_sci_plot = axes[0, 2].imshow(cc_sci_maps[cc_sci_SNR_closest_ind], origin='lower')

    axes[0, 2].set_title("Cross-correlation Sci Im Map")
    plt.colorbar(cc_map_plot, ax=axes[0,2])
    axes[0, 2].text(2, 95, "SNR={}".format(round(np.median(cc_sci_SNRs), 2)))

    plot_fl = "../plots/images_"
    if not r2_disk and not uniform_disk:
        title = "Real disk, incl={}, zodis={}, ".format(incl, zodis)
        plot_fl+= "realdisk_incl{}_zodis{}".format(incl, zodis)
    elif r2_disk and not uniform_disk:
        title = "1/r^2 disk, face-on, "
        plot_fl+= "r2disk_incl00_"
    elif uniform_disk and not r2_disk:
        title = "Uniform disk, face-on, "
        plot_fl += "unifdisk_incl00_"
    else:
        assert False, "Problem with plot title"
        
    if add_star:
        title += "star ON, "
        plot_fl += "starON_"
    else:
        title += "star OFF, "
        plot_fl += "starOFF_"
        
    if planet_noise:
        title += "planet noise ON"
        plot_fl += "plnoiseON"
    else:
        title += "planet noise OFF"
        plot_fl += "plnoiseOFF"
    
    plot_fl += ".png"
    fig.suptitle(title)
    print("Saving as {}".format(plot_fl))
    plt.savefig(plot_fl)
    plt.show()
    
if plot_median:
    plot_median_maps()



# print("\n\nmedian sci ref SNR", np.median(sci_ref_SNR))
print("median sub SNR before hipass", np.median(sub_SNRs))
print("median sub SNR after hipass", np.median(sub_SNRs_hipass))
print("median CC SNR", np.median(cc_SNRs))

print("median sci SNR", np.median(sci_SNRs))
print("median ref SNR", np.median(ref_SNRs))




