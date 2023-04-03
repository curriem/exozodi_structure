#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 08:25:16 2022

@author: mcurr
"""

import exozodi_functions as ezf
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy.interpolate import interp1d
import astropy.units as u
from scipy.ndimage import rotate


psf_dir = "/Users/mcurr/PACKAGES/coroSims/"

# parameters
tele = "LUVB"
pix_radius = 2
roll_angle = 90.

# nominal roll angles:
# LUV-A: 38.94244126898137
# LUV-B: 49.53983264223517


if tele == "LUVA":
    # load psf
    psf_offsets_fits = pyfits.open(psf_dir + "LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb/offax_psf_offset_list.fits")
    psfs_fits = pyfits.open(psf_dir + "LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb/offax_psf.fits")
elif tele == "LUVB":
    # load psf
    psf_offsets_fits = pyfits.open(psf_dir + "LUVOIR-B-VC6_timeseries/offax_psf_offset_list.fits")
    psfs_fits = pyfits.open(psf_dir + "LUVOIR-B-VC6_timeseries/offax_psf.fits")
psf_pixscale = psfs_fits[0].header["PIXSCALE"] # lam/D

# interpolate psf
fill = np.log(1e-100)
offax_offset_list = psf_offsets_fits[0].data
offax_list = psfs_fits[0].data
offax_list = offax_list[:, :-1, 1:]

inter_offaxis = interp1d(offax_offset_list[:, 0], np.log(offax_list), kind='cubic', axis=0, bounds_error=False, fill_value=fill) # interpolate in log-space to avoid negative values

    
if tele == "LUVB":
    # load an example image
    im_dir = "/Users/mcurr/PACKAGES/coroSims/LUVOIR-B_outputs/"
    im_dir += "scatteredlight-Mp_1.0-ap_1.0-incl_00-longitude_00-exozodis_1-distance_10/"
    sci_im_fits = pyfits.open(im_dir + "/PHN/sci_imgs.fits")
    sci_im = sci_im_fits[0].data[0, 0, 0]
    imsc = sci_im_fits[0].header["IMSC"] # lam/D
    imsz = sci_im_fits[0].header["IMSZ"] # pix
    diam = sci_im_fits[0].header["DIAM"]
    wave = sci_im_fits["WAVE"].data[0]
    rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im, imsc, diam, IWA_lamD=2.5, OWA_lamD=22., plotting=False)
elif tele == "LUVA":
    # load an example image
    im_dir = "/Users/mcurr/PACKAGES/coroSims/LUVOIR-A_outputs/"
    im_dir += "scatteredlight-Mp_1.0-ap_1.0-incl_00-longitude_00-exozodis_1-distance_10/"
    sci_im_fits = pyfits.open(im_dir + "/PHN/sci_imgs.fits")
    sci_im = sci_im_fits[0].data[0, 0, 0]
    imsc = sci_im_fits[0].header["IMSC"] # lam/D
    imsz = sci_im_fits[0].header["IMSZ"] # pix
    diam = sci_im_fits[0].header["DIAM"]
    wave = sci_im_fits["WAVE"].data[0]
    rotation_map, valid_mask, radius_map = ezf.construct_maps(sci_im, imsc, diam, IWA_lamD=8.5, OWA_lamD=22., plotting=False) # OWA changed from 26. for 5 pix radius case
        
    
    
print("Calculating matched filters...")    
# calculate psf for each psf pixel
Npix_i, Npix_j = rotation_map.shape

if (Npix_i % 2) == 0:
    center_i, center_j = Npix_j / 2. + 0.5, Npix_j / 2.+0.5
else:
    center_i, center_j = (Npix_j - 1)/2,(Npix_j-1)/2

matched_filter_datacube = np.empty((Npix_i, Npix_j, Npix_i, Npix_j))
matched_filter_single_datacube = np.empty((Npix_i, Npix_j, Npix_i, Npix_j))

matched_filter_metadata = np.empty((Npix_i, Npix_j, 2)) #rotation_angle, dist_from_center_lamD

counter = 0
for i in range(Npix_i):
    for j in range(Npix_j):


        # get the rotation angle for this pixel
        rotation_angle1 = rotation_map[i, j]

        # rotation angle for ADI counterpart
        rotation_angle2 = rotation_map[i, j] + roll_angle

        # see if the pixel is within valid range
        valid = valid_mask[i, j]

        if valid:

            # distances from pixel center to center of image
            x_1 = i  - center_i
            y_1 = j  - center_j

            # total distance from the pixel center to center of image
            dist_from_center = np.sqrt((x_1)**2 + (y_1)**2)
            dist_from_center_mas = dist_from_center * imsc
            dist_from_center_lamD = ezf.mas_to_lamD(dist_from_center_mas*u.mas, wave*u.um, diam*u.m)

            psf_raw = np.exp(inter_offaxis(dist_from_center_lamD))
            
# =============================================================================
#             plt.figure()
#             plt.imshow(psf_raw[110:130, 160:200], origin='lower')
#             plt.title(dist_from_center_lamD)
#             plt.show()
# =============================================================================
            

            # calculate the off-axis psf for this pixel
            psf1 = np.exp(rotate(np.log(psf_raw),-rotation_angle1, axes=(0, 1), reshape=False, mode='nearest', order=5))
            
            # downbin
            psf1 = ezf.downbin_psf(psf1, imsc, imsz, wave, diam, tele)
            
            # calculated psf for ADI counterpart
            psf2 = np.exp(rotate(np.log(psf_raw),-rotation_angle2, axes=(0, 1), reshape=False, mode='nearest', order=5))
            
            psf2 = ezf.downbin_psf(psf2, imsc, imsz, wave, diam, tele)
            
            
            
            x_2 = dist_from_center * np.sin(np.deg2rad(rotation_angle2))
            y_2 = dist_from_center * np.cos(np.deg2rad(rotation_angle2))
            i_2 = x_2 + center_i
            j_2 = y_2 + center_j


            i2_round = round(i_2)
            j2_round = round(j_2)

            psf_stamp1 = ezf.get_psf_stamp(psf1, i, j, pix_radius)
            psf_stamp2 = ezf.get_psf_stamp(psf2, round(i_2), round(j_2), pix_radius)

            psf_stamp1_norm = psf_stamp1 / np.max(psf_stamp1)
            psf_stamp2_norm = psf_stamp2 / np.max(psf_stamp2)


            matched_filter = np.zeros_like(psf1)
            matched_filter[i-pix_radius:i+pix_radius+1, j-pix_radius:j+pix_radius+1] = psf_stamp1_norm
            matched_filter[i2_round-pix_radius : i2_round+pix_radius+1, j2_round-pix_radius : j2_round+pix_radius+1] = -1*psf_stamp2_norm

            matched_filter_single = np.zeros_like(psf1)
            matched_filter_single[i-pix_radius:i+pix_radius+1, j-pix_radius:j+pix_radius+1] = psf_stamp1_norm


            matched_filter_datacube[i, j, :] = matched_filter
            matched_filter_metadata[i, j, :] = [rotation_angle1, dist_from_center_lamD]
            matched_filter_single_datacube[i, j, :] = matched_filter_single

            if counter % 1000 == 0:

                print("working on coords", i, j)
                print("RAW:", center_i, center_j + dist_from_center)
                print("IM1:", i, j)
                print("IM2:", i_2, j_2)
                print("IM2 round:", i2_round, j2_round)

                working_map = np.zeros_like(rotation_map, dtype=bool)
                working_map[i, j] = True


# =============================================================================
#                 plt.figure(figsize=(50,50))
#                 plt.title("{} {}".format(rotation_angle1, dist_from_center_lamD))
#                 plt.imshow(matched_filter, origin='lower')
#                 plt.axhline(i, color="white", linestyle="--")
#                 plt.axvline(j, color="white", linestyle="--")
#                 plt.axhline(i2_round, color="orange", linestyle="--")
#                 plt.axvline(j2_round, color="orange", linestyle="--")
# =============================================================================

            counter += 1


save_dir = "/Users/mcurr/PROJECTS/exozodi_structure/matched_filter_library/"

save_tag = "{}_rang{}_aprad{}".format(tele, round(roll_angle), pix_radius)


np.save(save_dir+"matched_filter_datacube_{}.npy".format(save_tag), matched_filter_datacube)
np.save(save_dir+"matched_filter_single_datacube_{}.npy".format(save_tag), matched_filter_single_datacube)

np.save(save_dir+"matched_filter_metadata_{}.npy".format(save_tag), matched_filter_metadata)
