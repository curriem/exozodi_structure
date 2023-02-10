# -*- coding: utf-8 -*-
import exozodi_functions as ezf
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt



# =============================================================================
# TODO:
#      1) vary high-pass filter size and aperture size -- this requires re-calculating 
#         the matched filter. Need to save different instances of matched filter 
#      2) choose a noise region that best reproduces the expected value--
#         get as close to the planet as possible. perhaps annulus around planet
#         test with 1/r^2 model 
#      3) apply these new methods to real disks. make plots
#      4) plot: measured/expected noise of hipass image vs. zodi level
#      5) plot: filter size vs. aperture size. colors for SNRs. different lines
#         for different zodis
#      6) plot: cc SNR achieved vs. zodi level
#      7) apply these methods to LUVOIR-B coronagraph data
#      8) start writing up paper
# DONE 9) figure out how many iterations are sufficient to get to right answer
#         (median changes by less than 1%?) DONE
#     10) Note: LUV-B matched filters look weird-- shift is off? 
#           --- it's not the order in the shift function. already tested.
#           --- the LUVB planet PSF that comes out of Jens' code is ugly, not perfect
#           --- like LUVA! Is this expected? Need to email Jens about this.
#           --- In any case, I think I got to the bottom of it.
# =============================================================================

# define some parameters
tele = "LUVA" # telescope


zodis = "10" # zodi level you want to work with
incl = "00"
longitude = "00"
pix_radius = 1
roll_angle = 90.

zodis_arr = ["1", "5", "10", "20", "50", "100"]
incl_arr = ["00", "30", "60", "90"]

matched_filter_dir = "/Users/mcurr/PROJECTS/exozodi_structure/matched_filter_library/"

matched_filter_fl = matched_filter_dir + "matched_filter_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), pix_radius)
matched_filter_datacube = np.load(matched_filter_fl)

matched_filter_single_fl = matched_filter_dir + "matched_filter_single_datacube_{}_rang{}_aprad{}.npy".format(tele, round(roll_angle), pix_radius)
matched_filter_single_datacube = np.load(matched_filter_single_fl)



if tele == "LUVA":
    planet_pos_lamD = 10.5
    planet_pos_mas = 100.26761414789404
    
if tele == "LUVB":
    planet_pos_lamD = 7.757018897752577 # lam/D
    planet_pos_mas = 100.
    matched_filter_datacube = np.load("/Users/mcurr/PACKAGES/coroSims//matched_filter_LUVB_datacube.npy")
    matched_filter_single_datacube = np.load("Users/mcurr/PACKAGES/coroSims//matched_filter_LUVA_single_datacube.npy")

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

sci_x = loc_of_planet_pix
sci_y = 0 

sci_signal_i = round(central_pixel)
sci_signal_j = round(central_pixel + loc_of_planet_pix)


aperture = ezf.get_psf_stamp(np.copy(sci_im), sci_signal_i, sci_signal_j, pix_radius) > 0


ref_x = loc_of_planet_pix * np.sin(np.deg2rad(roll_angle))
ref_y = loc_of_planet_pix * np.cos(np.deg2rad(roll_angle))

ref_signal_i = round(ref_x + central_pixel)
ref_signal_j = round(ref_y + central_pixel)





###############################################################################
add_noise = True
add_star = False
planet_noise = False
uniform_disk = True
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
while not converged:
    
    iterations += 1



#for n in range(num_iter):
    print(iterations)
    sci_im, ref_im, sci_planet_counts, ref_planet_counts = ezf.synthesize_images(im_dir, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, float(zodis), aperture,
                                       target_SNR=7, pix_radius=pix_radius,
                                       verbose=False, 
                                       add_noise=add_noise, 
                                       add_star=add_star, 
                                       planet_noise=planet_noise, 
                                       uniform_disk=uniform_disk,
                                       r2_disk=r2_disk)
    
    total_planet_counts = sci_planet_counts + ref_planet_counts
    
    # calculate maps
    rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im, imsc, diam, IWA_lamD=7.5, plotting=False)



    sci_im[~valid_mask] = np.nan
    ref_im[~valid_mask] = np.nan


    
    noise_region_radius = 10

    if ~planet_noise:
        force_signal = None #total_planet_counts # for just planet + uniform disk
        sci_force_signal = None #sci_planet_counts
        ref_force_signal = None #ref_planet_counts
    else:
        force_signal = None
        sci_force_signal = None
        ref_force_signal = None
        
    sci_SNR = ezf.calculate_SNR(sci_im, sci_signal_i, sci_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=r2_correct, force_signal=sci_force_signal)
    ref_SNR = ezf.calculate_SNR(ref_im, ref_signal_i, ref_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=r2_correct, force_signal=ref_force_signal)
    
    
    sub_im = sci_im - ref_im
    
    sub_SNR = ezf.calculate_SNR_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=r2_correct, force_signal=force_signal)
    
    
    ## CALCULATE OPTIMAL HIPASS FILTER SIZE
    ## remove known/unknown stuff... just using it to speed up code for now
    if optimal_filtersize_unknown:
        SNRs_filtersize = []
        filtersizes = np.arange(1, 20, 1)
        for filtersize in filtersizes:
            #print("Trying filtersize={}".format(filtersize))
            cc_map_test = ezf.calculate_cc_map(matched_filter_datacube, sub_im, valid_mask, hipass=True, filtersize=filtersize)
        
            cc_SNR_test = ezf.cc_SNR_known_loc(cc_map_test, sci_signal_i, sci_signal_j, pix_radius, roll_angle, aperture, central_pixel, noise_region_radius, r2_correct=r2_correct, mask_antisignal=True)
    
            SNRs_filtersize.append(cc_SNR_test)
            
        max_ind = np.argmax(SNRs_filtersize)
        optimal_filtersize = filtersizes[max_ind]
        optimal_filtersize_unknown = False
        

    
    if plot:
        # plot optimal filtersize
        plt.figure(figsize=(5,5))
        plt.plot(filtersizes, SNRs_filtersize)
        
        plt.axvline(optimal_filtersize, color="k", linestyle="--")
        plt.ylabel("CC SNR")
        plt.xlabel("Hipass filter size")
        
    
    sub_im_hipass = ezf.high_pass_filter(sub_im, filtersize=optimal_filtersize)
    
    sci_im_hipass = ezf.high_pass_filter(sci_im, filtersize=optimal_filtersize)
    
    sub_SNR_hipass = ezf.calculate_SNR_ADI(sub_im_hipass, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=r2_correct, force_signal=force_signal)
    
    #sci_ref_SNRs.append(sci_ref_SNR)
    
    
    
    # cross correlation
    cc_map_sci = ezf.calculate_cc_map(matched_filter_single_datacube, sci_im_hipass, valid_mask)
    cc_map = ezf.calculate_cc_map(matched_filter_datacube, sub_im_hipass, valid_mask)
    
    cc_SNR = ezf.cc_SNR_known_loc(cc_map, sci_signal_i, sci_signal_j, pix_radius, roll_angle, aperture, central_pixel, noise_region_radius, r2_correct=r2_correct, mask_antisignal=True)
    
    cc_SNR_sci = ezf.cc_SNR_known_loc(cc_map_sci, sci_signal_i, sci_signal_j, pix_radius, roll_angle, aperture, central_pixel, noise_region_radius, r2_correct=r2_correct, mask_antisignal=True)
    
    
    
    
    sci_SNRs.append(sci_SNR)
    ref_SNRs.append(ref_SNR)
    sub_SNRs.append(sub_SNR)
    sub_SNRs_hipass.append(sub_SNR_hipass)
    cc_SNRs.append(cc_SNR)
    cc_sci_SNRs.append(cc_SNR_sci)
    
    
    sci_ims.append(sci_im)
    ref_ims.append(ref_im)
    sub_ims.append(sub_im)
    sub_hipass_ims.append(sub_im_hipass)
    cc_sci_maps.append(cc_map_sci)
    cc_maps.append(cc_map)
    
    
    cc_median_vals.append(np.median(cc_SNRs))
    
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        
        sci_im_plot = axes[0, 0].imshow(np.log10(sci_im), origin='lower')
        axes[0, 0].set_title("Science Image")
        plt.colorbar(sci_im_plot, ax=axes[0,0])
        # axes[0, 0].text(2, 95, "SNR={}".format(round(sci_SNR, 2)))
        
        
        ref_im_plot = axes[1, 0].imshow(np.log10(ref_im), origin='lower')
        axes[1, 0].set_title("Reference Image")
        plt.colorbar(ref_im_plot, ax=axes[1,0])
        # axes[1, 0].text(2, 95, "SNR={}".format(round(ref_SNR, 2)))

        
        sub_im_plot = axes[0, 1].imshow(sub_im, origin='lower')
        axes[0, 1].set_title("Roll Subtracted Image")
        plt.colorbar(sub_im_plot, ax=axes[0, 1])
        # axes[0, 1].text(2, 95, "SNR={}".format(round(sub_SNR, 2)))

        
        sub_im_hipass_plot = axes[1, 1].imshow(sub_im_hipass, origin='lower')
        axes[1, 1].set_title("Hipass Roll Sub Image")
        plt.colorbar(sub_im_hipass_plot, ax=axes[1,1])
        # axes[1, 1].text(2, 95, "SNR={}".format(round(sub_SNR_hipass, 2)))
        
        cc_map_plot = axes[1, 2].imshow(cc_map, origin='lower')
        axes[1, 2].set_title("Cross-correlation Map")
        plt.colorbar(cc_map_plot, ax=axes[1,2])
        # axes[1, 2].text(2, 95, "SNR={}".format(round(cc_SNR, 2)))
        
        cc_map_sci_plot = axes[0, 2].imshow(cc_map_sci, origin='lower')
        axes[0, 2].set_title("Cross-correlation Sci Im Map")
        plt.colorbar(cc_map_plot, ax=axes[0,2])
        # axes[0, 2].text(2, 95, "SNR={}".format(round(cc_SNR_sci, 2)))

        plt.show()
        
# =============================================================================
#     print(cc_SNRs)
#     print("med:", np.median(cc_SNRs))
#     print("stds:", np.std(cc_SNRs))    
# =============================================================================
    

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

    if iterations > 10:
        

        if (frac_diff_med < 0.01) and (frac_diff_std < 0.01):
            print("Converged")
            convergence_counter += 1
            if convergence_counter == 10:
                converged = True
        else:
            # reset the convergence counter
            convergence_counter = 0

    if iterations == 1000:
        print("NOT CONVERGED: Iteration limit reached.")
        break
        

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



