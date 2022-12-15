# -*- coding: utf-8 -*-
import exozodi_functions as ezf
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt


# define some parameters
tele = "LUVA" # telescope


zodis = "100" # zodi level you want to work with
incl = "90"
longitude = "00"

zodis_arr = ["1", "5", "10", "20", "50", "100"]
incl_arr = ["00", "30", "60", "90"]

if tele == "LUVA":
    planet_pos_lamD = 10.5
    planet_pos_mas = 100.26761414789404
    roll_angle = 38.94244126898137
    matched_filter_datacube = np.load("/Users/mcurr/PACKAGES/coroSims//matched_filter_LUVA_datacube.npy")
    matched_filter_single_datacube = np.load("/Users/mcurr/PACKAGES/coroSims//matched_filter_LUVA_single_datacube.npy")
if tele == "LUVB":
    planet_pos_lamD = 7.757018897752577 # lam/D
    planet_pos_mas = 100.
    roll_angle = 49.53983264223517
    matched_filter_datacube = np.load("/Users/mcurr/PACKAGES/coroSims//matched_filter_LUVB_datacube.npy")
    matched_filter_single_datacube = np.load("Users/mcurr/PACKAGES/coroSims//matched_filter_LUVA_single_datacube.npy")

im_dir = "/Users/mcurr/PACKAGES/coroSims/LUVOIR-A_outputs/"
im_dir += "scatteredlight-Mp_1.0-ap_1.0-incl_{}-longitude_{}-exozodis_{}-distance_10/".format(incl, longitude, zodis)

# open an image just to get some information about it
sci_im_fits = pyfits.open(im_dir + "/PHN/sci_imgs.fits")
sci_im = sci_im_fits[0].data[0, 0, 0]
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

pix_radius = 1
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
r2_correct = True
# set noise seed
#np.random.seed(0)

#### r2_correct = True for everything except uniform disk


num_iter = 100

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

for n in range(num_iter):
    print(n)
    sci_im, ref_im, planet_signal = ezf.synthesize_images(im_dir, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, float(zodis), aperture,
                                       target_SNR=7, pix_radius=pix_radius,
                                       verbose=verbose, 
                                       add_noise=add_noise, 
                                       add_star=add_star, 
                                       planet_noise=planet_noise, 
                                       uniform_disk=uniform_disk,
                                       r2_disk=r2_disk)
    # calculate maps
    rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im, imsc, diam, IWA_lamD=7.5, plotting=False)



    sci_im[~valid_mask] = np.nan
    ref_im[~valid_mask] = np.nan


    
    noise_region_radius = 10

    sci_SNR = ezf.calculate_SNR(sci_im, sci_signal_i, sci_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=r2_correct)#, force_signal=72.3965585413765)
    ref_SNR = ezf.calculate_SNR(ref_im, ref_signal_i, ref_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=r2_correct)#, force_signal=66.99572433110286)
    
    if ~planet_noise:
        force_signal = planet_signal # for just planet + uniform disk
    else:
        force_signal = None
    
    sub_im = sci_im - ref_im
    
    sub_SNR = ezf.calculate_SNR_ADI(sub_im, sci_signal_i, sci_signal_j, ref_signal_i, ref_signal_j, valid_mask, aperture, noise_region_radius, r2_correct=r2_correct, force_signal=force_signal)
    
    
    ## CALCULATE OPTIMAL HIPASS FILTER SIZE
    SNRs_filtersize = []
    filtersizes = np.arange(1, 20, 1)
    for filtersize in filtersizes:
        #print("Trying filtersize={}".format(filtersize))
        cc_map_test = ezf.calculate_cc_map(matched_filter_datacube, sub_im, valid_mask, hipass=True, filtersize=filtersize)
    
        cc_SNR_test = ezf.cc_SNR_known_loc(cc_map_test, sci_signal_i, sci_signal_j, pix_radius, roll_angle, aperture, central_pixel, noise_region_radius, r2_correct=r2_correct, mask_antisignal=True)

        SNRs_filtersize.append(cc_SNR_test)
        
    max_ind = np.argmax(SNRs_filtersize)
    optimal_filtersize = filtersizes[max_ind]
    
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
    
    
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        
        sci_im_plot = axes[0, 0].imshow(np.log10(sci_im), origin='lower')
        axes[0, 0].set_title("Science Image")
        plt.colorbar(sci_im_plot, ax=axes[0,0])
        axes[0, 0].text(2, 95, "SNR={}".format(round(sci_SNR, 2)))
        
        
        ref_im_plot = axes[1, 0].imshow(np.log10(ref_im), origin='lower')
        axes[1, 0].set_title("Reference Image")
        plt.colorbar(ref_im_plot, ax=axes[1,0])
        axes[1, 0].text(2, 95, "SNR={}".format(round(ref_SNR, 2)))

        
        sub_im_plot = axes[0, 1].imshow(sub_im, origin='lower')
        axes[0, 1].set_title("Roll Subtracted Image")
        plt.colorbar(sub_im_plot, ax=axes[0, 1])
        axes[0, 1].text(2, 95, "SNR={}".format(round(sub_SNR, 2)))

        
        sub_im_hipass_plot = axes[1, 1].imshow(sub_im_hipass, origin='lower')
        axes[1, 1].set_title("Hipass Roll Sub Image")
        plt.colorbar(sub_im_hipass_plot, ax=axes[1,1])
        axes[1, 1].text(2, 95, "SNR={}".format(round(sub_SNR_hipass, 2)))
        
        cc_map_plot = axes[1, 2].imshow(cc_map, origin='lower')
        axes[1, 2].set_title("Cross-correlation Map")
        plt.colorbar(cc_map_plot, ax=axes[1,2])
        axes[1, 2].text(2, 95, "SNR={}".format(round(cc_SNR, 2)))
        
        cc_map_sci_plot = axes[0, 2].imshow(cc_map_sci, origin='lower')
        axes[0, 2].set_title("Cross-correlation Sci Im Map")
        plt.colorbar(cc_map_plot, ax=axes[0,2])
        axes[0, 2].text(2, 95, "SNR={}".format(round(cc_SNR_sci, 2)))

        plt.show()
        
    

if plot_median:
    median_cc_SNR = np.median(cc_SNRs)
    difference_array = np.absolute(cc_SNRs-median_cc_SNR)
    
    closest_ind = difference_array.argmin()


    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    
    sci_im_plot = axes[0, 0].imshow(np.log10(sci_ims[closest_ind]), origin='lower')
    axes[0, 0].set_title("Science Image")
    plt.colorbar(sci_im_plot, ax=axes[0,0])
    axes[0, 0].text(2, 95, "SNR={}".format(round(sci_SNRs[closest_ind], 2)))
    
    
    ref_im_plot = axes[1, 0].imshow(np.log10(ref_ims[closest_ind]), origin='lower')
    axes[1, 0].set_title("Reference Image")
    plt.colorbar(ref_im_plot, ax=axes[1,0])
    axes[1, 0].text(2, 95, "SNR={}".format(round(ref_SNRs[closest_ind], 2)))

    
    sub_im_plot = axes[0, 1].imshow(sub_ims[closest_ind], origin='lower')
    axes[0, 1].set_title("Roll Subtracted Image")
    plt.colorbar(sub_im_plot, ax=axes[0, 1])
    axes[0, 1].text(2, 95, "SNR={}".format(round(sub_SNRs[closest_ind], 2)))

    
    sub_im_hipass_plot = axes[1, 1].imshow(sub_hipass_ims[closest_ind], origin='lower')
    axes[1, 1].set_title("Hipass Roll Sub Image")
    plt.colorbar(sub_im_hipass_plot, ax=axes[1,1])
    axes[1, 1].text(2, 95, "SNR={}".format(round(sub_SNRs_hipass[closest_ind], 2)))
    
    cc_map_plot = axes[1, 2].imshow(cc_maps[closest_ind], origin='lower')
    axes[1, 2].set_title("Cross-correlation Map")
    plt.colorbar(cc_map_plot, ax=axes[1,2])
    axes[1, 2].text(2, 95, "SNR={}".format(round(cc_SNRs[closest_ind], 2)))
    
    cc_map_sci_plot = axes[0, 2].imshow(cc_sci_maps[closest_ind], origin='lower')
    axes[0, 2].set_title("Cross-correlation Sci Im Map")
    plt.colorbar(cc_map_plot, ax=axes[0,2])
    axes[0, 2].text(2, 95, "SNR={}".format(round(cc_sci_SNRs[closest_ind], 2)))

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


# print("\n\nmedian sci ref SNR", np.median(sci_ref_SNR))
print("median sub SNR before hipass", np.median(sub_SNRs))
print("median sub SNR after hipass", np.median(sub_SNRs_hipass))
print("median CC SNR", np.median(cc_SNRs))

print("median sci SNR", np.median(sci_SNRs))
# print("median ref SNR", np.median(ref_SNR))



