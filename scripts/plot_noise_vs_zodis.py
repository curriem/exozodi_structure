import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data as pandas dataframe

df = pd.read_csv('data.dat', sep=" ", header=0)

ap_sz = 3


print(df.to_string())

print(df["ap_sz"])

print(df[df["incl"].isin([0.])]["median_cc_SNR"])


zodis_arr = df.zodis.unique()
ap_sz_arr = df.ap_sz.unique()
filter_sz_arr = df.filter_sz.unique()


plt.figure(figsize=(8,6))

for incl in [0., 30., 60., 90]:


    # select vals
    incl_bool = df["incl"].isin([incl])
    filter_bool = df["filter_sz"].isin([10.])
    ap_sz_bool = df["ap_sz"].isin([ap_sz])
    tot_bool = incl_bool & ap_sz_bool & filter_bool
    
    
    print(df[tot_bool].to_string())
    
    measured_noise_before_hipass = df[tot_bool]["measured_noise_before_hipass"]
    expected_noise = df[tot_bool]["expected_noise"]
    zodis = df[tot_bool]["zodis"]
    
    plt.plot(zodis, measured_noise_before_hipass / expected_noise, label = incl)
    
plt.xlabel("Zodis")
plt.ylabel("measured/expected noise")
plt.legend(title="inclination")
plt.yscale("log")
    
    
    
    
    


