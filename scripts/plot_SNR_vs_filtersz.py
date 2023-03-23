# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("../../cg_high_res/plotting_scripts/miles_style.mplstyle")


# load data as pandas dataframe

noise_region = "circle"

df = pd.read_csv('data_{}.dat'.format(noise_region), sep=" ", header=0)

# =============================================================================
# print(df.to_string())
# 
# print(df["ap_sz"])
# 
# print(df[df["incl"].isin([0.])]["median_cc_SNR"])
# =============================================================================


zodis_arr = df.zodis.unique()
ap_sz_arr = df.ap_sz.unique()
filter_sz_arr = df.filter_sz_pix.unique()





for incl in [0., 30., 60., 90.]:
    fig1, axes1 = plt.subplots(2,3,figsize=(8, 6), sharey=True, sharex=True)

    fig2, axes2 = plt.subplots(2,3,figsize=(8, 6), sharey=True, sharex=True)
    
    fig3, axes3 = plt.subplots(2,3,figsize=(8, 6), sharey=False, sharex=True)
    
    fig4, axes4 = plt.subplots(2,3,figsize=(8, 6), sharey=False, sharex=True)

    fig5, axes5 = plt.subplots(2,3,figsize=(8, 6), sharey=False, sharex=True)


    ax1flt = axes1.flatten()
    ax2flt = axes2.flatten()
    ax3flt = axes3.flatten()
    ax4flt = axes4.flatten()
    ax5flt = axes5.flatten()




    for axis_i in range(len(ax1flt)):
        zodis = zodis_arr[axis_i]
        
        for ap_sz in ap_sz_arr:
            # select vals
            incl_bool = df["incl"].isin([incl])
            zodi_bool = df["zodis"].isin([zodis])
            ap_sz_bool = df["ap_sz"].isin([ap_sz])
            unif_disk_bool = df["uniform_disk"].isin([0])
            tot_bool = incl_bool & zodi_bool & ap_sz_bool  & unif_disk_bool
            
            
            # get median_cc vals
            median_cc_SNR = df[tot_bool]["median_cc_SNR_after_hipass"]
            SNR_after_hipass = df[tot_bool]["median_SNR_after_hipass"]
            median_signal = df[tot_bool]["median_measured_signal_after_hipass"]
            measured_noise = df[tot_bool]["measured_noise_after_hipass"]
            expected_noise = df[tot_bool]["expected_noise"]
            
            ax1flt[axis_i].plot(filter_sz_arr, median_cc_SNR, label=ap_sz)#, c=median_cc_SNR, cmap='plasma')
            ax2flt[axis_i].plot(filter_sz_arr, SNR_after_hipass, label=ap_sz)#, c=median_cc_SNR, cmap='plasma')
            ax3flt[axis_i].plot(filter_sz_arr, measured_noise/expected_noise, label=ap_sz)#, c=median_cc_SNR, cmap='plasma')
            ax4flt[axis_i].plot(filter_sz_arr, median_signal, label=ap_sz)#, c=median_cc_SNR, cmap='plasma')
            ax5flt[axis_i].plot(filter_sz_arr, measured_noise, label=ap_sz)#, c=median_cc_SNR, cmap='plasma')
            
            ax4flt[axis_i].axhline(1, color="k", linestyle=":")
            ax5flt[axis_i].axhline(1, color="k", linestyle=":")



        ax1flt[axis_i].set_title("{} zodis".format(round(zodis)))
        ax2flt[axis_i].set_title("{} zodis".format(round(zodis)))
        ax3flt[axis_i].set_title("{} zodis".format(round(zodis)))
        ax4flt[axis_i].set_title("{} zodis".format(round(zodis)))
        ax5flt[axis_i].set_title("{} zodis".format(round(zodis)))
        
        
        ax4flt[axis_i].set_yscale("log")
        ax5flt[axis_i].set_yscale("log")
        

        
    axes1[1,2].legend(title="aperture radius [pix]")
    axes2[1,2].legend(title="aperture radius [pix]")
    axes3[1,2].legend(title="aperture radius [pix]")
    axes4[1,2].legend(title="aperture radius [pix]")
    axes5[1,2].legend(title="aperture radius [pix]")

# =============================================================================
#     axes1[0,0].set_xscale("log")
#     axes2[0,0].set_xscale("log")
#     axes3[0,0].set_xscale("log")
# =============================================================================
    
    
    for i in range(3):
        axes1[1,i].set_xlabel("Hipass Filter Size [pix]")
        axes2[1,i].set_xlabel("Hipass Filter Size [pix]")
        axes3[1,i].set_xlabel("Hipass Filter Size [pix]")
        axes4[1,i].set_xlabel("Hipass Filter Size [pix]")
        axes5[1,i].set_xlabel("Hipass Filter Size [pix]")
        
    for i in range(2):
        axes1[i,0].set_ylabel("CC SNR")
        axes2[i,0].set_ylabel("SNR")
        axes3[i,0].set_ylabel("measured / expected noise")
        axes4[i,0].set_ylabel("measured signal")
        axes5[i,0].set_ylabel("measured noise")


        
    fig1.suptitle("Noise region: {}, {} deg inclination".format(noise_region, round(incl)))
    fig1.savefig("../plots/CCSNR_vs_HPFS_incl{}_{}.png".format(round(incl), noise_region))
    
    fig2.suptitle("Noise region: {}, {} deg inclination".format(noise_region, round(incl)))
    fig2.savefig("../plots/SNR_vs_HPFS_incl{}_{}.png".format(round(incl), noise_region))
    
    fig3.suptitle("Noise region: {}, {} deg inclination".format(noise_region, round(incl)))
    fig3.savefig("../plots/noise_vs_HPFS_incl{}_{}.png".format(round(incl), noise_region))
    
    
    fig4.suptitle("Noise region: {}, {} deg inclination".format(noise_region, round(incl)))
    fig5.suptitle("Noise region: {}, {} deg inclination".format(noise_region, round(incl)))

    
        

