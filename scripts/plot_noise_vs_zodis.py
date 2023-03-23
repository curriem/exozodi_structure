import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("../../cg_high_res/plotting_scripts/miles_style.mplstyle")

# load data as pandas dataframe

noise_region = "circle"

df = pd.read_csv('data_{}.dat'.format(noise_region), sep=" ", header=0)

ap_sz = 1.


print(df.to_string())
print(df["ap_sz"])

print(df[df["incl"].isin([0.])]["median_cc_SNR_after_hipass"])


zodis_arr = df.zodis.unique()
ap_sz_arr = df.ap_sz.unique()
filter_sz_arr = df.filter_sz_pix.unique()



fig, axes = plt.subplots(2, 1, figsize=(6.5,4), sharey=True)
# models
for i, incl in enumerate([0., 30., 60., 90]):

    incl_label = str(int(incl)) + r"$^\circ$"

    # select vals
    incl_bool = df["incl"].isin([incl])
    if incl == 90.:
        filter_bool = df["filter_sz_pix"].isin([10.])
    else:
        filter_bool = df["filter_sz_pix"].isin([10.])
    ap_sz_bool = df["ap_sz"].isin([ap_sz])
    model_bool = df["uniform_disk"].isin([0.])
    tot_bool = model_bool & incl_bool & ap_sz_bool & filter_bool
    
    
    print(df[tot_bool].to_string())
    
    measured_noise_before_hipass = df[tot_bool]["measured_noise_before_hipass"]
    measured_noise_after_hipass = df[tot_bool]["measured_noise_after_hipass"]

    expected_noise = df[tot_bool]["expected_noise"]
    
    measured_noise_before_hipass_out = df[tot_bool]["median_measured_noise_before_hipass_out"]
    measured_noise_after_hipass_out = df[tot_bool]["median_measured_noise_after_hipass_out"]

    expected_noise_out = df[tot_bool]["expected_noise_out"]
    
    zodis = df[tot_bool]["zodis"]
    
    axes[0].plot(zodis, measured_noise_before_hipass / expected_noise, color="C{}".format(i), linestyle="--")
    axes[1].plot(zodis, measured_noise_before_hipass_out / expected_noise_out, color="C{}".format(i), linestyle="--")
    
    axes[0].plot(zodis, measured_noise_after_hipass / expected_noise, label = incl_label+ " disk model", color="C{}".format(i))
    axes[1].plot(zodis, measured_noise_after_hipass_out / expected_noise_out, label = incl_label+ " disk model", color="C{}".format(i))


# uniform disk
incl_bool = df["incl"].isin([0.])
filter_bool = df["filter_sz_pix"].isin([10.])
ap_sz_bool = df["ap_sz"].isin([ap_sz])
model_bool = df["uniform_disk"].isin([1.])
tot_bool = incl_bool & filter_bool & ap_sz_bool & model_bool
zodis = df[tot_bool]["zodis"]
measured_noise_before_hipass_unif = df[tot_bool]["measured_noise_before_hipass"]
measured_noise_after_hipass_unif = df[tot_bool]["measured_noise_after_hipass"]

expected_noise_unif = df[tot_bool]["expected_noise"]
axes[0].plot(zodis, measured_noise_before_hipass_unif/expected_noise_unif, color="C6", label="uniform disk", linestyle="-")
axes[1].plot(zodis, measured_noise_after_hipass_unif/expected_noise_unif, color="C6", label="uniform disk", linestyle="-")


plt.xlabel("Zodis")
axes[0].set_ylabel(r"N$_\mathrm{meas}$/N$_\mathrm{expt}$")
axes[1].set_ylabel(r"N$_\mathrm{meas}$/N$_\mathrm{expt}$")
axes[0].legend(loc="upper left", fontsize=6, framealpha=0.7)
axes[1].legend(ncol=1, loc="upper left", fontsize=6)

axes[0].set_yscale("log")
axes[1].set_yscale("log")



axes[0].text(0.5, 0.95, 'At Planet Location',
        horizontalalignment='center',
        verticalalignment='top',
        transform=axes[0].transAxes, fontsize=12)

axes[1].text(0.5, 0.95, 'Outside Resonant Structure',
        horizontalalignment='center',
        verticalalignment='top',
        transform=axes[1].transAxes, fontsize=12)

for i in range(2):
    ax2 = axes[i].twinx()
    ax2.plot(np.NaN, np.NaN, ls="--",
             label='before hipass', c='black')
    ax2.plot(np.NaN, np.NaN, ls="-",
             label='after hipass', c='black')
    ax2.get_yaxis().set_visible(False)

    ax2.legend(loc="upper right", fontsize=6)
    axes[i].set_ylim(0.6, 2e4)
    
    axes[i].set_xscale("log")


plt.suptitle("Noise region: {}".format(noise_region))
plt.savefig("../plots/noise_vs_zodis_{}.png".format(noise_region))
    
    
    


