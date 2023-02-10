#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:19:54 2023

@author: mcurr
"""

# REMEMBER TO conda activate coro

import numpy as np

import coronagraph
import detector
import util
import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, correlate2d
from scipy.ndimage import rotate, zoom, shift
import glob
import os
from scipy.interpolate import interp1d


def calculate_positions(disk_model, ap, incl, longitude, dist, pixelscale, plot=False):

    
    incl = incl.to(u.radian)
    longitude = longitude.to(u.radian)

    #pixelsize_pc = 2 * dist * np.tan(pixelscale / 2)
    #pixelsize_pc = 2 * dist * np.tan(pixelscale / 2)

    pixelsize_pc = dist * np.sin(pixelscale)

    pixelsize_au = pixelsize_pc.to(u.au)
    pix_per_au = 1 / pixelsize_au

    # distance from star to planet in pixels
    r_planet = pix_per_au * ap

    x_npix, y_npix = disk_model.shape
    
    img_center = (x_npix - 1)/2

    # star coordinates 
    x_star, y_star = img_center, img_center

    # planet coordinates
    x_plan = x_star + r_planet * np.cos(longitude)
    y_plan = y_star + r_planet * np.sin(longitude) * np.cos(incl)
    if plot:
        plt.figure()
        plt.imshow(disk_model)
        plt.scatter(x_star, y_star, color="lime",s=50, marker="*")
        plt.scatter(x_plan, y_plan, color="lime", s=50, marker="o")
        plt.show()
    return x_star, y_star, x_plan, y_plan


def load_disk(disk_fl):
    disk_fits = pyfits.open(disk_fl)
    disk = disk_fits[0].data
    wavelengths = disk_fits[1].data
    fstar = disk_fits[2].data

    return disk, wavelengths, fstar


def downbin_image(image, binsize, pixelscale):
    nx, ny = image.shape
    assert nx % binsize == 0
    assert ny % binsize == 0
    
    new_nx, new_ny = int(nx/binsize), int(ny/binsize)

    new_image = np.empty((new_nx, new_ny))
    
    for x in range(new_nx):
        for y in range(new_ny):
            new_image[x, y] = np.sum(image[binsize*x:binsize*x+binsize, binsize*y:binsize*y+binsize])
    
    new_pixelscale = pixelscale * binsize
    return new_image, new_pixelscale

def find_max_psf_radius(loc, coro, airy_min=2, plotting=False):
    psf = coro.inter_offaxis(loc)
    psf_pixscale = coro.imsc # lam/D
    center_i, center_j = np.unravel_index(psf.argmax(), psf.shape)
    psf_slice_i = psf[center_i]
    psf_slice_j = psf[:, center_j]
    
    # do first slice
    local_maxima_i = argrelextrema(psf_slice_i, np.greater)[0]
    local_minima_i = argrelextrema(psf_slice_i, np.less)[0]
    
    minima_gt_max = local_minima_i[local_minima_i > center_j]
    minima_lt_max = local_minima_i[local_minima_i < center_j]
    
    upper_r_i = minima_gt_max[airy_min-1]
    lower_r_i = minima_lt_max[-1*airy_min]
    
    # do second slice
    local_maxima_j = argrelextrema(psf_slice_j, np.greater)[0]
    local_minima_j = argrelextrema(psf_slice_j, np.less)[0]
    
    minima_gt_max = local_minima_j[local_minima_j > center_i]
    minima_lt_max = local_minima_j[local_minima_j < center_i]
    
    upper_r_j = minima_gt_max[airy_min-1]
    lower_r_j = minima_lt_max[-1*airy_min]
    
    
    
    if plotting:
        offset_i = len(psf_slice_i)/2
        offset_j = len(psf_slice_j)/2
        plt.figure()
        plt.plot((np.arange(len(psf_slice_i))-offset_i)*psf_pixscale, psf_slice_i)
        plt.axvline((center_j-offset_i)*psf_pixscale, linestyle=":", color="k")
        plt.axvline((lower_r_i-offset_i)*psf_pixscale, linestyle=":", color="blue")
        plt.axvline((upper_r_i-offset_i)*psf_pixscale, linestyle=":", color="red")
        plt.figure()
        plt.plot((np.arange(len(psf_slice_j)) - offset_j) *psf_pixscale, psf_slice_j)
        plt.axvline((center_i-offset_i)*psf_pixscale, linestyle=":", color="k")
        plt.axvline((lower_r_j-offset_i)*psf_pixscale, linestyle=":", color="blue")
        plt.axvline((upper_r_j-offset_i)*psf_pixscale, linestyle=":", color="red")
    
    
    radii_of_psf = [center_j-lower_r_i, upper_r_i-center_j, center_i-lower_r_j, upper_r_j-center_i]
    return np.max(radii_of_psf) * psf_pixscale # in lam/D
    

# def find_max_psf_radius(loc, coro, plotting=False):
#     psf = coro.inter_offaxis(loc)
#     psf_pixscale = coro.imsc # lam/D
#     center_i, center_j = np.unravel_index(psf.argmax(), psf.shape)
#     psf_slice_i = psf[center_i]
#     psf_slice_j = psf[:, center_j]
    
#     def find_radius(psf_slice, center, which_rad):
#         test_ind = 0
#         prev_r = center - test_ind
#         while True:
#             test_ind +=1
#             if which_rad == "lower":
#                 radius = center - test_ind
#             elif which_rad == "upper":
#                 radius = center + test_ind
#             if psf_slice[radius] > psf_slice[prev_r]:
#                 break
#             elif radius < 0:
#                 break
#             elif radius > 239:
#                 break
#             else:
#                 prev_r = radius
#         return radius
    
#     upper_r_i = find_radius(psf_slice_i, center_j, "upper")
#     lower_r_i = find_radius(psf_slice_i, center_j, "lower")
    
#     upper_r_j = find_radius(psf_slice_j, center_i, "upper")
#     lower_r_j = find_radius(psf_slice_j, center_i, "lower")
#     if plotting:
#         offset_i = len(psf_slice_i)/2
#         offset_j = len(psf_slice_j)/2
#         plt.figure()
#         plt.plot((np.arange(len(psf_slice_i))-offset_i)*psf_pixscale, psf_slice_i)
#         plt.axvline((center_j-offset_i)*psf_pixscale, linestyle=":", color="k")
#         plt.axvline((lower_r_i-offset_i)*psf_pixscale, linestyle=":", color="blue")
#         plt.axvline((upper_r_i-offset_i)*psf_pixscale, linestyle=":", color="red")
#         plt.figure()
#         plt.plot((np.arange(len(psf_slice_j)) - offset_j) *psf_pixscale, psf_slice_j)
#         plt.axvline((center_i-offset_i)*psf_pixscale, linestyle=":", color="k")
#         plt.axvline((lower_r_j-offset_i)*psf_pixscale, linestyle=":", color="blue")
#         plt.axvline((upper_r_j-offset_i)*psf_pixscale, linestyle=":", color="red")
        
#     radii_of_psf = [center_j-lower_r_i, upper_r_i-center_j, center_i-lower_r_j, upper_r_j-center_i]
#     return np.max(radii_of_psf) * psf_pixscale # in lam/D


def mas_to_lamD(sep_mas, lam, D):
    # sep_mas: planet--star separation in mas
    # lam: wl of observation in um
    # D: diam of telescope in m
    
    # returns lamD: separation in lam/D
    lam = lam.to(u.m)
    sep_lamD = (D/lam) * (1/u.radian) * sep_mas.to(u.radian)
    return sep_lamD

def lamD_to_mas(lamD_sep, lam, D):
    # sep_lamD: planet--star separation in lamD
    # lam: wl of observation in um
    # D: diam of telescope in m
    # returns sep_mas: separation in mas
    
    lam = lam.to(u.m)
    sep_mas = (lamD_sep * (lam / D) * u.radian).to(u.mas)
    
    return sep_mas

def mas_to_AU(sep_mas, dist, pixelscale):
    # sep_mas: planet--star separation in mas
    # dist: distance to system in pc
    # pixelscale: pixel scale of detector in mas
    
    pixelsize_pc = dist * np.sin(pixelscale)

    pixelsize_au = pixelsize_pc.to(u.au)
    
    r_planet_pixels = sep_mas /pixelscale
    
    sep_AU = pixelsize_au * r_planet_pixels
    return sep_AU

def AU_to_mas(sep_AU, dist, pixelscale):
    # sep_AU: planet--star separation in AU
    # dist: distance to system in pc
    # pixelscale: pixel scale of detector in mas
    pixelsize_pc = dist * np.sin(pixelscale)

    pixelsize_au = pixelsize_pc.to(u.au)
    
    r_planet_pixels = sep_AU / pixelsize_au
    
    sep_mas = r_planet_pixels*pixelscale
    
    return sep_mas

def AU_to_lamD(sep_AU, dist, pixelscale, lam, D):
    # sep_AU: planet--star separation in AU
    # dist: distance to system in pc
    # pixelscale: pixel scale of detector in mas
    # lam: wl of observation in um
    # D: diam of telescope in m
    
    sep_mas = AU_to_mas(sep_AU, dist, pixelscale)
    
    sep_lamD = mas_to_lamD(sep_mas, lam, D)
    
    return sep_lamD

def lamD_to_AU(sep_lamD, dist, pixelscale, lam, D):
    # sep_lamD: planet--star separation in lam/D
    # dist: distance to system in pc
    # pixelscale: pixel scale of detector in mas
    # lam: wl of observation in um
    # D: diam of telescope in m
    
    sep_mas = lamD_to_mas(sep_lamD, lam, D)
    
    sep_au = mas_to_AU(sep_mas, dist, pixelscale)
    
    return sep_au

def calculate_roll_angle(loc_of_target_lamD, coro):
    # calculates the minimum rotation angle you need to separate planet psf cores
    # core is defined as a circle around psf with radius of first minimum of airy function
    
    # psf if slightly squished in y direction, so take maximum radius of first minimum:
    psf_radius_lamD = find_max_psf_radius(loc_of_target_lamD, coro, plotting=False)

    rot_angle = 2 * np.arcsin(psf_radius_lamD/loc_of_target_lamD) * u.radian
    
    return rot_angle.to(u.deg)

def interpolate_disk(disk_data, disk_wls, disk_fstar, fill=np.log(1e-100)):
    inter_disk = interp1d(disk_wls, np.log(disk_data), kind="cubic", axis=0, bounds_error=False, fill_value=fill)
    inter_fstar = interp1d(disk_wls, np.log(disk_fstar), kind="cubic", axis=0, bounds_error=False, fill_value=fill)

    return inter_disk, inter_fstar

class scene():
    
    def __init__(self, time, Ntime, pixscale, xystar, fstar, Nplanets, xyplanet, fplanet, disk, wave, Nwave, angdiam):
        self.time = time
        self.Ntime = Ntime
        self.pixscale = pixscale
        self.xystar = xystar
        self.fstar = fstar
        self.Nplanets = Nplanets
        self.xyplanet = xyplanet
        self.fplanet = fplanet
        self.disk = disk
        self.wave = wave
        self.Nwave = Nwave
        self.angdiam = angdiam
        

overwrite = True
nofwdscat = False
Nobs = 100

disk_dir = "/Users/mcurr/PROJECTS/idl_projects/dustmap/dustmap_outputs/scattered_light_runs/"
if nofwdscat:
    disk_dir = "/Users/mcurr/PROJECTS/idl_projects/dustmap/dustmap_outputs/no_forward_scattering/"




def generate_ims(tele, Mp, ap, incl, long, zodis, dist, wavelength):
    
   

    if tele == "LUVOIR-A":
        diam = 12. # m
        iwa = 8.5 # lambda/D, for LUV-A
        owa = 26. # lambda/D, for LUV-A
        output_fl = "../data/LUVOIR-A_outputs/"
        if nofwdscat:
            output_fl = "nofwdscat_outputs/"
        cdir = '/Users/mcurr/PACKAGES/coroSims/LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb/'

        
    elif tele == "LUVOIR-B":
        diam = 8.
        iwa = 2.5 # lambda/D, for LUV-B
        owa = 13. # lambda/D, for LUV-B
        output_fl = "../data/LUVOIR-B_outputs/"
        cdir = '/Users/mcurr/PACKAGES/coroSims/LUVOIR-B-VC6_timeseries/'

    
    tag = "scatteredlight-Mp_{}-ap_{}-incl_{}-longitude_{}-exozodis_{}-distance_{}".format(Mp, ap, incl, long, zodis, dist)
    if nofwdscat:
        tag = "no_fwd_scatt-Mp_{}-ap_{}-incl_{}-longitude_{}-exozodis_{}-distance_{}".format(Mp, ap, incl, long, zodis, dist)
    
    
    disk_fl = disk_dir + tag+".fits"
    
    print(disk_fl)
    
    
    fdir = output_fl
    odir = output_fl

    # convert params from strings to floats

    
    Mp = float(Mp)
    ap = float(ap)
    incl = float(incl)
    long = float(long)
    dist = float(dist)
    zodis = float(zodis)

    
    
    # load the disk
    disks, disk_wls, disk_fstar = load_disk(disk_fl)
    
    # interpolate disk
    
    inter_disk, inter_fstar = interpolate_disk(disks, disk_wls, disk_fstar)

    disk = np.exp(inter_disk(wavelength))
    disk_naninds = np.where(np.isnan(disk))

    disk[disk_naninds] = 0.
    
    fstar_interp = np.exp(inter_fstar(wavelength))

    wave = np.array([wavelength])
    

    # downbin the disk
    pixelscale = 1.07429586587 * u.mas
    if tele == "LUVOIR-A":
        disk_binsize = 2
    elif tele == "LUVOIR-B":
        disk_binsize = 3
    downbinned_disk, new_pixelscale = downbin_image(disk, disk_binsize, pixelscale)

    # get rid of zeros in disk -> replace with second smallest value
    second_smallest = sorted(np.unique(downbinned_disk.flatten()))[1]
    downbinned_disk[np.where(downbinned_disk == 0.)] = second_smallest

    # get the planet and star positions
    x_star, y_star, x_plan, y_plan = calculate_positions(downbinned_disk, ap*u.AU, incl*u.deg, long*u.deg, dist*u.pc, new_pixelscale, plot=True)
    
    
    print(downbinned_disk.shape)
    
    print(x_star, y_star, x_plan, y_plan )
    
    
    
    
    # initialize the coronagraph 
    coro = coronagraph.coro(cdir)


    # set up my own scene to bypass exovista scene
    time = np.array([0.])
    Ntime = 1
    pixscale = new_pixelscale.value
    xystar = np.array([[[x_star, y_star]]])
    fstar = np.array([[[fstar_interp]]])
    Nplanets = 1
    xyplanet = np.array([[[x_plan, y_plan]]])
    fplanet = np.array([[[4.44285286e-09]]])
    disk = np.array([[downbinned_disk]]) * fstar
    Nwave = 1
    angdiam = 0.465 # angular diameter of the star [mas]
    
    
    
    planet_pos_mas_actual = np.sqrt((x_plan - x_star)**2 + (y_plan - y_star)**2) * new_pixelscale
    planet_pos_lamD_actual = mas_to_lamD(planet_pos_mas_actual, wave[0]*u.um, diam*u.m)
    
    # round to closest 0.25
    #planet_pos_lamD_forced = round(planet_pos_lamD_actual.value*4)/4
   # print("lamD", planet_pos_lamD_actual, planet_pos_lamD_forced)
    # round to closest 0.5 lam/D in final image space
    imsc_final = 0.5*(wavelength * 1e-6 / (0.9*diam) * u.radian).to(u.mas)
    print(planet_pos_mas_actual, planet_pos_lamD_actual)
    print(imsc_final)
    def nearest(n, x):
        asdf = n % x > x // 2
        return n + (-1)**(1 - asdf) * abs(x * asdf - n % x)
    planet_pos_mas_forced = nearest(planet_pos_mas_actual.value, imsc_final.value)

#     planet_pos_mas_forced = round(planet_pos_mas_actual.value * imsc_final.value) / imsc_final.value
    #planet_pos_mas_forced = lamD_to_mas(planet_pos_lamD_forced, wave[0]*u.um, diam*u.m)
    #print("mas", planet_pos_mas_actual, planet_pos_mas_forced)
    print("planet_pos_mas_forced", planet_pos_mas_forced)
    print("imsc final", imsc_final)
    x_plan_forced = np.sqrt((planet_pos_mas_forced/new_pixelscale.value)**2 - (y_plan-y_star)**2) + x_star
    xyplanet = np.array([[[x_plan_forced, y_plan]]])
    print(xyplanet)
    
    

    
    
    coro.scene = scene(time, Ntime, pixscale, xystar, fstar, Nplanets, xyplanet, fplanet, disk, wave, Nwave, angdiam)
    
    coro.angdiam = angdiam
    coro.diam = diam
    coro.dist = dist
    coro.vmag = 4.83 # of the star
    coro.odir = odir

    # define aperture
    papt = xyplanet[0,0] - xystar[0,0]
    rapt = 0.8 # lambda/D



    # Unit conversion factor from Jy to ph/s.
    # Flux F_nu is given in Jy = 10^(-26)*W/Hz/m^2.
    # Flux F_lambda = F_nu*c/lambda^2.
    # Photon energy E = h*c/lambda.
    # Count rate in ph/s = 10^(-26)*F_nu*A*dl/h/lambda*T.
    tp = np.array([coro.insttp]*coro.scene.Nwave)
    area = np.pi*coro.diam**2/4.*(1.-coro.fobs) # m^2 (effective mirror area)
    dl = coro.bp*wave*1e-6 # m
    coro.phot = 1e-26*area*dl/6.626e-34/(wave*1e-6)*tp # ph/s for source with 1 Jy

    coro.set_pang(0.) # deg

    # calculate the roll angle based on planet position
    planet_pos_mas = np.sqrt((x_plan_forced - x_star)**2 + (y_plan - y_star)**2) * new_pixelscale
    print("Planet position (mas)", planet_pos_mas)
    planet_pos_lamD = mas_to_lamD(planet_pos_mas, wave[0]*u.um, 0.9*diam*u.m)
    
    phi_roll = calculate_roll_angle(planet_pos_lamD.value, coro)
    print("Roll angle:", phi_roll)
    print("Planet position (lam/D)", planet_pos_lamD)
    #phi_roll = 30. * u.deg
    coro.phi_roll = phi_roll
    coro.planet_pos_lamD = planet_pos_lamD
    
    tag += "-rang_{}".format(round(phi_roll.value)) 
    coro.name = tag
    
    
    # make the science and reference coronagraph images
    sci = coro.sim_sci(add_star=True,
                       add_plan=True,
                       add_disk=True,
                       tag='sci',
                       save_all=True)
    ref = coro.sim_ref(rang=phi_roll.value,
                       add_star=True,
                       add_plan=True,
                       add_disk=True,
                       tag='ref',
                       save_all=True)

    # convert coronagraph images to detector images
    det = detector.det(iwa=iwa,
                       owa=owa)

    det.nlimgs(tag,
               odir,
               tags=['sci_imgs', 'sci_star', 'sci_plan', 'sci_disk', 'ref_imgs', 'ref_star', 'ref_plan', 'ref_disk'],
               #tags = ["sci_plan"],
               overwrite=overwrite)

# =============================================================================
#     tint, cr_star, cr_plan, cr_disk, cr_detn = det.tint(tag,
#                                                         odir,
#                                                         tag='sci_imgs',
#                                                         papt=papt, # pix
#                                                         rapt=rapt, # lambda/D
#                                                         time_comp=time[0], # yr
#                                                         wave_comp=wave[0], # micron
#                                                         snr_targ=7.,
#                                                         path_star=odir+tag+'/DET/sci_star.fits',
#                                                         path_plan=odir+tag+'/DET/sci_plan.fits',
#                                                         path_disk=odir+tag+'/DET/sci_disk.fits',
#                                                         detn=False,
#                                                         fdir=fdir)
# 
# =============================================================================
# =============================================================================
#     # add photon noise
#     det.pnimgs(tag,
#                odir,
#                tags=['sci_imgs', 'ref_imgs'],
#                tint=tint, # s
#                time_comp=time[0], # yr
#                wave_comp=wave[0], # micron
#                Nobs=Nobs,
#                overwrite=overwrite)
# =============================================================================

    # subtract science and reference images
    #subtract_sci_ref(os.getcwd()+"/"+odir+name)
    
    return coro, det





def generate_ims_nozodi(tele, Mp, ap, incl, long, dist, wavelength):
    
    zodis = "1"
   

    if tele == "LUVOIR-A":
        diam = 12. # m
        iwa = 8.5 # lambda/D, for LUV-A
        owa = 26. # lambda/D, for LUV-A
        output_fl = "LUVOIR-A_outputs/"
        if nofwdscat:
            output_fl = "nofwdscat_outputs/"
        cdir = '/Users/mcurr/PACKAGES/coroSims/LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb/'

        
    elif tele == "LUVOIR-B":
        diam = 8.
        iwa = 2.5 # lambda/D, for LUV-B
        owa = 13. # lambda/D, for LUV-B
        output_fl = "LUVOIR-B_outputs/"
        cdir = '/Users/mcurr/PACKAGES/coroSims/LUVOIR-B-VC6_timeseries/'

    
    tag = "scatteredlight-Mp_{}-ap_{}-incl_{}-longitude_{}-exozodis_{}-distance_{}".format(Mp, ap, incl, long, zodis, dist)
    if nofwdscat:
        tag = "no_fwd_scatt-Mp_{}-ap_{}-incl_{}-longitude_{}-exozodis_{}-distance_{}".format(Mp, ap, incl, long, zodis, dist)
    
    disk_fl = disk_dir + tag+".fits"
    
    print(disk_fl)
    
    fdir = output_fl
    odir = output_fl

    # convert params from strings to floats
    Mp_str = Mp
    ap_str = ap
    incl_str = incl
    long_str = long
    dist_str = dist
    Mp = float(Mp)
    ap = float(ap)
    incl = float(incl)
    long = float(long)
    dist = float(dist)
    zodis = float(zodis)

    
    
    # load the disk
    disks, disk_wls, disk_fstar = load_disk(disk_fl)
    
    # interpolate disk
    
    inter_disk, inter_fstar = interpolate_disk(disks, disk_wls, disk_fstar)

    disk = np.exp(inter_disk(wavelength))
    disk_naninds = np.where(np.isnan(disk))

    disk[disk_naninds] = 0.
    
    fstar_interp = np.exp(inter_fstar(wavelength))

    wave = np.array([wavelength])
    

    # downbin the disk
    pixelscale = 1.07429586587 * u.mas
    if tele == "LUVOIR-A":
        disk_binsize = 2
    elif tele == "LUVOIR-B":
        disk_binsize = 3
    downbinned_disk, new_pixelscale = downbin_image(disk, disk_binsize, pixelscale)

    # get rid of zeros in disk -> replace with second smallest value
    second_smallest = sorted(np.unique(downbinned_disk.flatten()))[1]
    downbinned_disk[np.where(downbinned_disk == 0.)] = second_smallest

    # get the planet and star positions
    x_star, y_star, x_plan, y_plan = calculate_positions(downbinned_disk, ap*u.AU, incl*u.deg, long*u.deg, dist*u.pc, new_pixelscale, plot=True)
    
    
    print(downbinned_disk.shape)
    
    print(x_star, y_star, x_plan, y_plan )
    
    
    
    
    # initialize the coronagraph 
    coro = coronagraph.coro(cdir)


    # set up my own scene to bypass exovista scene
    time = np.array([0.])
    Ntime = 1
    pixscale = new_pixelscale.value
    xystar = np.array([[[x_star, y_star]]])
    fstar = np.array([[[fstar_interp]]])
    Nplanets = 1
    xyplanet = np.array([[[x_plan, y_plan]]])
    fplanet = np.array([[[4.44285286e-09]]])
    disk = np.array([[downbinned_disk]]) * fstar
    Nwave = 1
    angdiam = 0.465 # angular diameter of the star [mas]
    
    
    
    planet_pos_mas_actual = np.sqrt((x_plan - x_star)**2 + (y_plan - y_star)**2) * new_pixelscale
    planet_pos_lamD_actual = mas_to_lamD(planet_pos_mas_actual, wave[0]*u.um, diam*u.m)
    
    # round to closest 0.25
    #planet_pos_lamD_forced = round(planet_pos_lamD_actual.value*4)/4
   # print("lamD", planet_pos_lamD_actual, planet_pos_lamD_forced)
    # round to closest 0.5 lam/D in final image space
    imsc_final = 0.5*(wavelength * 1e-6 / (0.9*diam) * u.radian).to(u.mas)
    
    def nearest(n, x):
        asdf = n % x > x // 2
        return n + (-1)**(1 - asdf) * abs(x * asdf - n % x)
    planet_pos_mas_forced = nearest(planet_pos_mas_actual.value, imsc_final.value)

#     planet_pos_mas_forced = round(planet_pos_mas_actual.value * imsc_final.value) / imsc_final.value
    #planet_pos_mas_forced = lamD_to_mas(planet_pos_lamD_forced, wave[0]*u.um, diam*u.m)
    #print("mas", planet_pos_mas_actual, planet_pos_mas_forced)
    
    x_plan_forced = np.sqrt((planet_pos_mas_forced/new_pixelscale.value)**2 - (y_plan-y_star)**2) + x_star
    xyplanet = np.array([[[x_plan_forced, y_plan]]])
    
    
    
    zodis = "0"
    tag = "scatteredlight-Mp_{}-ap_{}-incl_{}-longitude_{}-exozodis_{}-distance_{}".format(Mp_str, ap_str, incl_str, long_str, zodis, dist_str)
   
    
    
    coro.scene = scene(time, Ntime, pixscale, xystar, fstar, Nplanets, xyplanet, fplanet, disk, wave, Nwave, angdiam)

    name = tag
    coro.name = name
    coro.angdiam = angdiam
    coro.diam = diam
    coro.dist = dist
    coro.vmag = 4.83 # of the star
    coro.odir = odir

    # define aperture
    papt = xyplanet[0,0] - xystar[0,0]
    rapt = 0.8 # lambda/D



    # Unit conversion factor from Jy to ph/s.
    # Flux F_nu is given in Jy = 10^(-26)*W/Hz/m^2.
    # Flux F_lambda = F_nu*c/lambda^2.
    # Photon energy E = h*c/lambda.
    # Count rate in ph/s = 10^(-26)*F_nu*A*dl/h/lambda*T.
    tp = np.array([coro.insttp]*coro.scene.Nwave)
    area = np.pi*coro.diam**2/4.*(1.-coro.fobs) # m^2 (effective mirror area)
    dl = coro.bp*wave*1e-6 # m
    coro.phot = 1e-26*area*dl/6.626e-34/(wave*1e-6)*tp # ph/s for source with 1 Jy

    coro.set_pang(0.) # deg

    # calculate the roll angle based on planet position
    planet_pos_mas = np.sqrt((x_plan_forced - x_star)**2 + (y_plan - y_star)**2) * new_pixelscale
    print("Planet position (mas)", planet_pos_mas)
    planet_pos_lamD = mas_to_lamD(planet_pos_mas, wave[0]*u.um, 0.9*diam*u.m)
    
    phi_roll = calculate_roll_angle(planet_pos_lamD.value, coro)
    print("Roll angle:", phi_roll)
    print("Planet position (lam/D)", planet_pos_lamD)
    coro.phi_roll = phi_roll
    coro.planet_pos_lamD = planet_pos_lamD
    
    
    # make the science and reference coronagraph images
    sci = coro.sim_sci(add_star=True,
                       add_plan=True,
                       add_disk=False,
                       tag='sci',
                       save_all=True)
    ref = coro.sim_ref(rang=phi_roll.value,
                       add_star=True,
                       add_plan=True,
                       add_disk=False,
                       tag='ref',
                       save_all=True)

    # convert coronagraph images to detector images
    det = detector.det(iwa=iwa,
                       owa=owa)

    det.nlimgs(name,
               odir,
               tags=['sci_imgs', 'sci_star', 'sci_plan', 'ref_imgs'],
               overwrite=overwrite)

    tint, cr_star, cr_plan, cr_disk, cr_detn = det.tint(name,
                                                        odir,
                                                        tag='sci_imgs',
                                                        papt=papt, # pix
                                                        rapt=rapt, # lambda/D
                                                        time_comp=time[0], # yr
                                                        wave_comp=wave[0], # micron
                                                        snr_targ=7.,
                                                        path_star=odir+name+'/DET/sci_star.fits',
                                                        path_plan=odir+name+'/DET/sci_plan.fits',
                                                        #path_disk=odir+name+'/DET/sci_disk.fits',
                                                        detn=False,
                                                        fdir=fdir)

    # add photon noise
    det.pnimgs(name,
               odir,
               tags=['sci_imgs', 'ref_imgs'],
               tint=tint, # s
               time_comp=time[0], # yr
               wave_comp=wave[0], # micron
               Nobs=Nobs,
               overwrite=overwrite)

    # subtract science and reference images
    #subtract_sci_ref(os.getcwd()+"/"+odir+name)
    
    return coro, det


# DISK RUNS

Mp = "1.0"
ap = "1.0"
incl = "00"
long = "00"
dist = "10"
zodis = "1"
wavelength=0.5

coro, det = generate_ims("LUVOIR-A", Mp, ap, incl, long, zodis, dist, wavelength)


do_this = True
wavelength=0.5
if do_this:
    Mp = "1.0"
    ap = "1.0"
    incl_arr = ["00", "30", "60", "90"]
    long_arr = ["00", "30"][:1]
    dist_arr = ["10", "15"][:1]
    zodi_arr = ["1", "5", "10", "20", "50", "100"]

    #coro = coronagraph.coro(cdir)

    for incl in incl_arr:
        for long in long_arr:
            for dist in dist_arr:
                for zodis in zodi_arr:
                    #pass
                    print("Running Mp={}, ap={}, incl={}, long={}, dist={}, zodi={}".format(Mp, ap, incl, long, dist, zodis))
                    coro, det = generate_ims("LUVOIR-A", Mp, ap, incl, long, zodis, dist, wavelength)


assert False

def downbin_psf(psf, imsz, wave, diam, tele):
    
    if tele == "LUVA":
        shift_order = 1
    elif tele == "LUVB":
        shift_order = 0
    
    rad2mas = 180./np.pi*3600.*1000.

    imsc2 = 0.5*0.5e-6/(0.9*diam)*rad2mas # mas
    
    # Compute wavelength-dependent zoom factor.
    fact = 0.25*wave * 1e-6 /diam*rad2mas/imsc2

    norm = np.sum(psf)

    # Scale image to imsc.
    temp = np.exp(zoom(np.log(psf), fact, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
    
    temp *= norm/np.sum(temp) # ph/s
    
    # Center image so that (imsz-1)/2 is center.
    if (((temp.shape[0] % 2 == 0) and (imsz % 2 != 0)) or ((temp.shape[0] % 2 != 0) and (imsz % 2 == 0))):
        temp = np.pad(temp, ((0, 1), (0, 1)), mode='edge')
        temp = np.exp(shift(np.log(temp), (0.5, 0.5), order=shift_order)) # interpolate in log-space to avoid negative values
        temp = temp[1:-1, 1:-1]
        
    # Crop image to imsz.
    if (temp.shape[0] > imsz):
        nn = (temp.shape[0]-imsz)//2
        temp = temp[nn:-nn, nn:-nn]
    else:
        nn = (imsz-temp.shape[0])//2
        temp = np.pad(temp, ((nn, nn), (nn, nn)), mode='edge')
        
    return temp

psf = np.exp(coro.inter_offaxis(10.5))
print(psf.shape)
plt.figure(figsize=(20,20))
plt.imshow(psf)
plt.axhline(120)
plt.axvline(147)

psf = pyfits.open("/Users/mcurr/PACKAGES/coroSims/LUVOIR-A_outputs/scatteredlight-Mp_1.0-ap_1.0-incl_00-longitude_00-exozodis_1-distance_10/RAW/sci_plan.fits")
psf = psf[0].data[0,0]
print(psf.shape)

psf_downbinned = downbin_psf(psf, 101, 0.5, 12., "LUVA")

plt.figure(figsize=(20,20))
plt.imshow(psf)
plt.axhline(120)
plt.axvline(147)

# NO DISK RUN

Mp = "1.0"
ap = "1.0"
incl = "00"
long = "00"
dist = "10"
#zodis = "0"
wavelength=0.5
coro, det = generate_ims_nozodi("LUVOIR-A", Mp, ap, incl, long, dist, wavelength)


# CHECK PLANET FLUX AS A FUNCTION OF ap

do_this_test = False
if do_this_test:
    
    iwa_rad = 8.5 * (0.5*u.um).to(u.m).value / 12 * u.radian # as
    iwa_mas = iwa_rad.to(u.mas)
    iwa_as = iwa_mas.to(u.arcsec)
    iwa_pc = dist * np.sin(iwa_as) *u.pc
    iwa_au = iwa_pc.to(u.AU)

    
    list_of_aps = np.arange(0., 1., 0.01) * u.AU
    plan_fluxes = []
    for ap_n in list_of_aps:
        # get the planet and star positions
        x_star, y_star, x_plan, y_plan = calculate_positions(coro.scene.disk[0, 0, :], ap_n, incl*u.deg, long*u.deg, dist*u.pc, coro.scene.pixscale*u.mas, plot=False)
        coro.scene.xystar = np.array([[[x_star, y_star]]]) # x and y positions of the star in pixels. middle of scene.
        coro.scene.xyplanet = np.array([[[x_plan, y_plan]]]) # position of planet in pixels

        plan = coro.add_plan(tag="test", save=False)

        plan_flux = np.sum(plan)
        plan_fluxes.append(plan_flux)

    plt.figure()
    plt.plot(list_of_aps, plan_fluxes)
    plt.ylabel("flux of planet scene")
    plt.xlabel("planet distance from star [AU]")
    plt.axvline(iwa_au.value, color="k", linestyle=":", label="IWA")
    
    
# Check magnitude of dust at 1AU for 1 zodi
## Should be ~22mag/arcsec^2

Mp = "1.0"
ap = "1.0"
incl = "00"
long = "00"
dist = "10"
zodis = "1"
wavelength=0.54
tag = "scatteredlight-Mp_{}-ap_{}-incl_{}-longitude_{}-exozodis_{}-distance_{}".format(Mp, ap, incl, long, zodis, dist)
disk_fl = disk_dir + tag+".fits"
disk_data, disk_wls, disk_fstar = load_disk(disk_fl)

print("Finding closest wavelength to", wavelength, "um ....")
closest_ind = (np.abs(disk_wls - wavelength)).argmin()
print("\t-> using", disk_wls[closest_ind], "um")

fstar_Vband = disk_fstar[closest_ind] # Jy
disk_Vband = disk_data[closest_ind] * fstar_Vband


# downbin the disk
pixelscale = 1.07429586587 * u.mas
downbinned_disk, new_pixelscale = downbin_image(disk_Vband, 2, pixelscale)

dist = float(dist)
pixelsize_pc = dist * np.sin(new_pixelscale) * u.pc

pixelsize_au = pixelsize_pc.to(u.au)
pix_per_au = 1 / pixelsize_au 

Npix_i, Npix_j = downbinned_disk.shape
center_pix = Npix_i/2
one_au_pix = pix_per_au.value

anulus_rad = 1 # AU
anulus_width = 0.1 # AU
central_r = 1 / pixelsize_au * anulus_rad
inner_r = central_r - 1 / pixelsize_au * anulus_width
outer_r = central_r + 1 / pixelsize_au * anulus_width
inner_r = inner_r.value
outer_r = outer_r.value

anulus_mask = np.zeros((Npix_i, Npix_j), dtype=bool)
for i_pix in range(Npix_i):
    for j_pix in range(Npix_j):
        dist_from_center =  np.sqrt(float(i_pix - center_pix)**2 + float(j_pix - center_pix)**2)
        if (dist_from_center > inner_r) & (dist_from_center < outer_r):
            anulus_mask[i_pix, j_pix] = True

plt.figure()
plt.imshow(downbinned_disk)

one_au_dust = downbinned_disk*anulus_mask

plt.figure()
plt.imshow(downbinned_disk*anulus_mask)

zeropoint_flux = 3.636e-20 * u.erg / u.cm**2 / u.s / u.Hz   # flux of vega in V band from Bessel 1998
zeropoint_flux = zeropoint_flux.to(u.Jy)

mag = -2.5 * np.log10(np.sum(one_au_dust) / zeropoint_flux.value)

area_mas2 = new_pixelscale**2 * np.sum(anulus_mask)
area_as2 = area_mas2.to(u.arcsec**2)


surface_brightness = mag + 2.5*np.log10(area_as2.value)


F0 = 3781

SB_JY = F0 * 10**(-0.4*surface_brightness)

print("mag:", mag)
print("flux:", np.sum(one_au_dust))
print("area_as2", area_as2)
print("surface brightness:", surface_brightness, "mag/arcsec^2")
print("sum one_au_dust", np.sum(one_au_dust))
print("SB_JY", SB_JY)

                       


# Loop over disk params

do_this = True
wavelength=0.5
if do_this:
    Mp = "1.0"
    ap = "1.0"
    incl_arr = ["00", "30", "60", "90"]
    long_arr = ["00", "30"]
    dist_arr = ["10", "15"]
    zodi_arr = ["1", "5", "10", "20", "50", "100"]

    #coro = coronagraph.coro(cdir)

    for incl in incl_arr:
        for long in long_arr:
            for dist in dist_arr:
                for zodis in zodi_arr:
                    #pass
                    print("Running Mp={}, ap={}, incl={}, long={}, dist={}, zodi={}".format(Mp, ap, incl, long, dist, zodis))
                    coro, det = generate_ims("LUVOIR-A", Mp, ap, incl, long, zodis, dist, wavelength)







