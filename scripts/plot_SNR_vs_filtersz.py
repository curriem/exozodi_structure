# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data as pandas dataframe

df = pd.read_csv('data.dat', sep=" ", header=0)

print(df.to_string())

print(df["ap_sz"])

print(df[df["incl"].isin([0.])]["median_cc_SNR"])


zodis_arr = df.zodis.unique()
ap_sz_arr = df.ap_sz.unique()
filter_sz_arr = df.filter_sz.unique()






for incl in [0., 30., 60., 90.]:
    fig1, axes1 = plt.subplots(2,3,figsize=(8, 6), sharey=True, sharex=True)

    fig2, axes2 = plt.subplots(2,3,figsize=(8, 6), sharey=True, sharex=True)

    ax1flt = axes1.flatten()
    ax2flt = axes2.flatten()

    for axis_i in range(len(ax1flt)):
        zodis = zodis_arr[axis_i]
        
        for ap_sz in ap_sz_arr:
            # select vals
            incl_bool = df["incl"].isin([incl])
            zodi_bool = df["zodis"].isin([zodis])
            ap_sz_bool = df["ap_sz"].isin([ap_sz])
            tot_bool = incl_bool & zodi_bool & ap_sz_bool 
            
            # get median_cc vals
            median_cc_SNR = df[tot_bool]["median_cc_SNR"]
            median_cc_SNR_before_hipass = df[tot_bool]["median_cc_SNR_before_hipass"]
            
            ax1flt[axis_i].plot(filter_sz_arr, median_cc_SNR, label=ap_sz)#, c=median_cc_SNR, cmap='plasma')
            ax2flt[axis_i].plot(filter_sz_arr, median_cc_SNR/median_cc_SNR_before_hipass, label=ap_sz)#, c=median_cc_SNR, cmap='plasma')

        ax1flt[axis_i].set_title("{} zodis".format(round(zodis)))
        ax2flt[axis_i].set_title("{} zodis".format(round(zodis)))
        
    axes1[1,2].legend(title="aperture radius [pix]")
    axes2[1,2].legend(title="aperture radius [pix]")
    
    for i in range(3):
        axes1[1,i].set_xlabel("Hipass Filter Size")
        axes2[1,i].set_xlabel("Hipass Filter Size")
        
    for i in range(2):
        axes1[i,0].set_ylabel("CC SNR")
        axes2[i,0].set_ylabel("CC SNR after / before hipass")
        
    fig1.suptitle("{} deg inclination".format(round(incl)))
    fig1.savefig("../plots/CCSNR_vs_HPFS_incl{}.png".format(round(incl)))
    
    fig2.suptitle("{} deg inclination".format(round(incl)))
    fig2.savefig("../plots/CCSNR2_vs_HPFS_incl{}.png".format(round(incl)))
    
    
        

