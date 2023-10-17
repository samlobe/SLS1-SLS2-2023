#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# from autocorrelation.py
# V300-S305
#     jR2R3       50 ns equilibration, 1055 uncorrelated samples
#     jR2R3_P301L 50 ns equilibration, 1615 uncorrelated samples
# K298-Q307
#     jR2R3       50 ns equilibration, 704  uncorrelated samples
#     jR2R3_P301L 50 ns equilibration, 1051 uncorrelated samples

# import data
pinch_jR2R3 = pd.read_csv('pair_distances_jR2R3.csv')['Distance: VAL300 and SER305'].values
pinch_jR2R3_P301L = pd.read_csv('pair_distances_jR2R3_P301L.csv')['Distance: VAL300 and SER305'].values
clamp_jR2R3 = pd.read_csv('pair_distances_jR2R3.csv')['Distance: LYS298 and GLN307'].values
clamp_jR2R3_P301L = pd.read_csv('pair_distances_jR2R3_P301L.csv')['Distance: LYS298 and GLN307'].values
jR2R3_len = len(pinch_jR2R3)
jR2R3_P301L_len = len(pinch_jR2R3_P301L)

# bootstrap jR2R3's data
clamp_jR2R3_filtered_boots = []
bootstrap_samples = 1000
for i in tqdm(range(bootstrap_samples)):
    # get 704 random indices with replacement
    random_indices = np.random.randint(0,jR2R3_len,704)
    # get the values at those indices
    pinch_jR2R3_boot = pinch_jR2R3[random_indices]
    clamp_jR2R3_boot = clamp_jR2R3[random_indices]

    # filter out values in the pinching phase: 0.4<clamp<1.8 ; 0.4<pinch<0.7
    clamp_jR2R3_filtered_boot = []
    for pinch,clamp in zip(pinch_jR2R3_boot,clamp_jR2R3_boot):
        if 0.4<pinch<0.7 and 0.4<clamp<1.8:
          clamp_jR2R3_filtered_boot.append(clamp)
    clamp_jR2R3_filtered_boots.append(clamp_jR2R3_filtered_boot)

# bootstrap jR2R3_P301L
clamp_jR2R3_P301L_filtered_boots = []
bootstrap_samples = 1000
for i in tqdm(range(bootstrap_samples)):
    # get 1615 random indices with replacement
    random_indices = np.random.randint(0,jR2R3_P301L_len,1615)
    # get the values at those indices
    pinch_jR2R3_P301L_boot = pinch_jR2R3_P301L[random_indices]
    clamp_jR2R3_P301L_boot = clamp_jR2R3_P301L[random_indices]

    # filter out values in the pinching phase: 0.4<clamp<0.7 ; 0.4<pinch<1.45
    clamp_jR2R3_P301L_filtered_boot = []
    for pinch,clamp in zip(pinch_jR2R3_P301L_boot,clamp_jR2R3_P301L_boot):
        if 0.4<pinch<0.7 and 0.4<clamp<1.8:
            clamp_jR2R3_P301L_filtered_boot.append(clamp)
    clamp_jR2R3_P301L_filtered_boots.append(clamp_jR2R3_P301L_filtered_boot)

#%% histogram to get probabilities in each bootstrapped set
# get the bin edges that we'll plot
bins = np.arange(0.4,1.85,0.075)
# get the bin mids
bin_mids = (bins[1:]+bins[:-1])/2
# for each bootstrap sample in jR2R3, store the histogram heights for these bins
clamp_jR2R3_filtered_boots_hist = []
for clamp_jR2R3_filtered_boot in clamp_jR2R3_filtered_boots:
    hist,bin_edges = np.histogram(clamp_jR2R3_filtered_boot,bins=bins)
    clamp_jR2R3_filtered_boots_hist.append(hist/len(clamp_jR2R3_filtered_boot))
clamp_jR2R3_filtered_boots_hist = np.array(clamp_jR2R3_filtered_boots_hist)
# for each bootstrap sample in jR2R3_P301L, store the histogram heights for these bins
clamp_jR2R3_P301L_filtered_boots_hist = []
for clamp_jR2R3_P301L_filtered_boot in clamp_jR2R3_P301L_filtered_boots:
    hist,bin_edges = np.histogram(clamp_jR2R3_P301L_filtered_boot,bins=bins)
    clamp_jR2R3_P301L_filtered_boots_hist.append(hist/len(clamp_jR2R3_P301L_filtered_boot))
clamp_jR2R3_P301L_filtered_boots_hist = np.array(clamp_jR2R3_P301L_filtered_boots_hist)
# for each bin, order the heights from lowest to highest
clamp_jR2R3_sorted_hist = np.sort(clamp_jR2R3_filtered_boots_hist, axis=0)
clamp_jR2R3_P301L_sorted_hist = np.sort(clamp_jR2R3_P301L_filtered_boots_hist, axis=0)
# add 0.00001 to all the probabilities so that the log doesn't blow up when converting to free energies
clamp_jR2R3_sorted_hist[clamp_jR2R3_sorted_hist==0] = 0.00001
clamp_jR2R3_P301L_sorted_hist[clamp_jR2R3_P301L_sorted_hist==0] = 0.00001

#%% sort and determine confidence intervals for each bin
confidence_interval = 67 #percent
lower_percentile = (100-confidence_interval)/2
upper_percentile = 100-lower_percentile

clamp_jR2R3_lower_bounds = np.percentile(clamp_jR2R3_sorted_hist, lower_percentile, axis=0)
clamp_jR2R3_upper_bounds = np.percentile(clamp_jR2R3_sorted_hist, upper_percentile, axis=0)

clamp_jR2R3_P301L_lower_bounds = np.percentile(clamp_jR2R3_P301L_sorted_hist, lower_percentile, axis=0)
clamp_jR2R3_P301L_upper_bounds = np.percentile(clamp_jR2R3_P301L_sorted_hist, upper_percentile, axis=0)

clamp_jR2R3_mean = np.percentile(clamp_jR2R3_sorted_hist, 50, axis=0)
clamp_jR2R3_P301L_mean = np.percentile(clamp_jR2R3_P301L_sorted_hist, 50, axis=0)
#%% plot the confidence intervals
plt.fill_between(bin_mids, clamp_jR2R3_lower_bounds, clamp_jR2R3_upper_bounds, color='tab:orange', alpha=0.2)
plt.plot(bin_mids, clamp_jR2R3_mean, 'tab:orange', label='jR2R3')

plt.fill_between(bin_mids, clamp_jR2R3_P301L_lower_bounds, clamp_jR2R3_P301L_upper_bounds, color='tab:blue', alpha=0.2)
plt.plot(bin_mids, clamp_jR2R3_P301L_mean, 'tab:blue', label='jR2R3 P301L')

plt.xlabel('Distance: V300-S305 (nm)', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.legend(fontsize=14)
plt.ylim(bottom=0)
plt.show()

#%% convert to free energies and replot
kb = 0.0083144621 #kJ/mol/K
T = 300 #K

clamp_jR2R3_mean_free = -kb*T*np.log(clamp_jR2R3_mean)
clamp_jR2R3_P301L_mean_free = -kb*T*np.log(clamp_jR2R3_P301L_mean)

clamp_jR2R3_lower_bounds_free = -kb*T*np.log(clamp_jR2R3_lower_bounds)
clamp_jR2R3_upper_bounds_free = -kb*T*np.log(clamp_jR2R3_upper_bounds)

clamp_jR2R3_P301L_lower_bounds_free = -kb*T*np.log(clamp_jR2R3_P301L_lower_bounds)
clamp_jR2R3_P301L_upper_bounds_free = -kb*T*np.log(clamp_jR2R3_P301L_upper_bounds)

# subtract so that the minimum free energy of each is 0
clamp_jR2R3_mean_free -= np.min(clamp_jR2R3_mean_free)
clamp_jR2R3_P301L_mean_free -= np.min(clamp_jR2R3_P301L_mean_free)

clamp_jR2R3_lower_bounds_free -= np.min(clamp_jR2R3_lower_bounds_free)
clamp_jR2R3_upper_bounds_free -= np.min(clamp_jR2R3_upper_bounds_free)

clamp_jR2R3_P301L_lower_bounds_free -= np.min(clamp_jR2R3_P301L_lower_bounds_free)
clamp_jR2R3_P301L_upper_bounds_free -= np.min(clamp_jR2R3_P301L_upper_bounds_free)

# replot
plt.fill_between(bin_mids, clamp_jR2R3_lower_bounds_free, clamp_jR2R3_upper_bounds_free, color='tab:orange', alpha=0.2)
plt.plot(bin_mids, clamp_jR2R3_mean_free, 'tab:orange', label='jR2R3')

plt.fill_between(bin_mids, clamp_jR2R3_P301L_lower_bounds_free, clamp_jR2R3_P301L_upper_bounds_free, color='tab:blue', alpha=0.2)
plt.plot(bin_mids, clamp_jR2R3_P301L_mean_free, 'tab:blue', label='jR2R3 P301L')

plt.title('Unclamping Free Energy',fontsize=14)
plt.xlabel('Distance: K298-Q307 (nm)', fontsize=14)
plt.ylabel('Free Energy (kJ/mol)', fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(bottom=0,top=10)
plt.xlim(left=0.45,right=1.1)

# save the figure
plt.savefig('unclamp_1D_FE_landscape.png',dpi=300,bbox_inches='tight')
plt.show()

#%%

