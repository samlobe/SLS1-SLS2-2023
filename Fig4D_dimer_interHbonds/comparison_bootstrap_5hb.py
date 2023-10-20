#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# equilibration time (from integrating autocorrelation function, trying multiple equilibration times and maximizing uncorrelated samples)
equil_jR2R3 = 110 #ns
equil_jR2R3_P301L = 80  #ns

# # using pymbar.timeseries.detectEquilibration
# equil_jR2R3 = 216 #ns 
# equil_jR2R3_P301L = 310  #ns

# load alpha carbon pairwise distances data
dist_dir = '../pair_distances/dimers'
columns_jR2R3 = ['Distance: LYS298 and GLN307','Distance: VAL300 and SER305']
columns_jR2R3_P301L = ['Distance: LYS298 and GLN307','Distance: VAL300 and SER305']
df_jR2R3A = pd.read_csv(f'{dist_dir}/jR2R3_monA.csv',usecols=columns_jR2R3).sort_index()[equil_jR2R3*100:]
df_jR2R3B = pd.read_csv(f'{dist_dir}/jR2R3_monB.csv',usecols=columns_jR2R3).sort_index()[equil_jR2R3*100:]
df_jR2R3_P301LA = pd.read_csv(f'{dist_dir}/jR2R3_P301L_monA.csv',usecols=columns_jR2R3_P301L).sort_index()[equil_jR2R3_P301L*100:]
df_jR2R3_P301LB = pd.read_csv(f'{dist_dir}/jR2R3_P301L_monB.csv',usecols=columns_jR2R3_P301L).sort_index()[equil_jR2R3_P301L*100:]

# load the number of intermolecular hydrogen bond per frame (created with  `gmx hbond ...` with mainchain+H selected for each monomer)
data_jR2R3_hbonds = np.loadtxt('jR2R3_hbnum.xvg',comments=['@','#'])[equil_jR2R3*100:,1]
data_jR2R3_P301L_hbonds = np.loadtxt('jR2R3_P301L_hbnum.xvg',comments=['@','#'])[equil_jR2R3_P301L*100:,1]

# num independent (from integrating autocorrelation function)
indep_jR2R3 = 1274 
indep_jR2R3_P301L = 1139

# using pymbar.timeseries.detectEquilibration
# indep_jR2R3 = 772
# indep_jR2R3_P301L = 605

#%% BOOTSTRAP FOR jR2R3

bootstrap_num = 500
data_bootstrapped = np.zeros(bootstrap_num)
bins = [0,4,np.inf]

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR2R3_hbonds),indep_jR2R3)
    boot_jR2R3_hbonds = data_jR2R3_hbonds[indices]
    boot_df_jR2R3A = df_jR2R3A.values[indices]
    boot_df_jR2R3B = df_jR2R3B.values[indices]
    data_bootstrapped[k] = np.histogram(boot_jR2R3_hbonds,bins=[5,np.inf],
                                        density=False)[0]/len(indices)

hbond_frac_confidence = np.zeros(3)
confidence = 0.90
sorted_frac = np.sort(data_bootstrapped)
il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
hbond_frac_confidence = low, mean, up
hbond_confidence_jR2R3 = pd.DataFrame(np.reshape(hbond_frac_confidence,(1,3)),index=['jR2R3'],
                                columns=[f'{int(50-(confidence*100/2))}%','50%',f'{int(50+(confidence*100/2))}%'])
#%% BOOTSTRAP FOR jR2R3_P301L
bootstrap_num = 500
data_bootstrapped = np.zeros(bootstrap_num)
bins = [0,4,np.inf]

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR2R3_P301L_hbonds),indep_jR2R3_P301L)
    boot_jR2R3_P301L_hbonds = data_jR2R3_P301L_hbonds[indices]
    boot_df_jR2R3_P301LA = df_jR2R3_P301LA.values[indices]
    boot_df_jR2R3_P301LB = df_jR2R3_P301LB.values[indices]
    data_bootstrapped[k] = np.histogram(boot_jR2R3_P301L_hbonds,bins=[5,np.inf],
                                        density=False)[0]/len(indices)

hbond_frac_confidence = np.zeros(3)
confidence = 0.90
sorted_frac = np.sort(data_bootstrapped)
il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
hbond_frac_confidence = low, mean, up
hbond_confidence_jR2R3_P301L = pd.DataFrame(np.reshape(hbond_frac_confidence,(1,3)),index=['jR2R3_P301L'],
                                columns=[f'{int(50-(confidence*100/2))}%','50%',f'{int(50+(confidence*100/2))}%'])


#%% NOW jR2R3  AGAIN BUT FILTERING FOR AT LEAST ONE MONOMER
# IN THE LOWER LEFT OF THE ENERGY LANDSCAPE (r298-307 < 0.7A); (r300-305 < 0.8A)
bootstrap_num = 500
data_bootstrapped = np.zeros(bootstrap_num)
bins = [0,4,np.inf]

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR2R3_hbonds),indep_jR2R3)
    boot_jR2R3_hbonds = data_jR2R3_hbonds[indices]
    boot_df_jR2R3A = df_jR2R3A.values[indices]
    boot_df_jR2R3B = df_jR2R3B.values[indices]

    mask_zip1A   = boot_df_jR2R3A[:,0] < 0.7
    mask_pinch1A = boot_df_jR2R3A[:,1] < 0.8
    mask_zip1B   = boot_df_jR2R3B[:,0] < 0.7
    mask_pinch1B = boot_df_jR2R3B[:,1] < 0.8
    mask_1A = list(np.logical_and(mask_zip1A, mask_pinch1A))
    mask_1B = list(np.logical_and(mask_zip1B, mask_pinch1B))
    mask_jR2R3_intra = list(np.logical_or(mask_1A, mask_1B))
    new_jR2R3_hbonds = boot_jR2R3_hbonds[mask_jR2R3_intra]

    data_bootstrapped[k] = np.histogram(new_jR2R3_hbonds,bins=[5,np.inf],
                                        density=False)[0]/len(indices)

hbond_frac_confidence = np.zeros(3)
confidence = 0.90
sorted_frac = np.sort(data_bootstrapped)
il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
hbond_frac_confidence = low, mean, up
hbond_confidence_jR2R3_left = pd.DataFrame(np.reshape(hbond_frac_confidence,(1,3)),index=['jR2R3_left'],
                                columns=[f'{int(50-(confidence*100/2))}%','50%',f'{int(50+(confidence*100/2))}%'])

#%% NOW jR2R3_P301L  AGAIN BUT FILTERING FOR AT LEAST ONE MONOMER
# IN THE LOWER LEFT OF THE ENERGY LANDSCAPE (r298-307 < 0.7A); (r300-305 < 0.8A)
bootstrap_num = 500
data_bootstrapped = np.zeros(bootstrap_num)
bins = [0,4,np.inf]

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR2R3_P301L_hbonds),indep_jR2R3_P301L)
    boot_jR2R3_P301L_hbonds = data_jR2R3_P301L_hbonds[indices]
    boot_df_jR2R3_P301LA = df_jR2R3_P301LA.values[indices]
    boot_df_jR2R3_P301LB = df_jR2R3_P301LB.values[indices]

    mask_zip2A   = boot_df_jR2R3_P301LA[:,0] < 0.7
    mask_pinch2A = boot_df_jR2R3_P301LA[:,1] < 0.8
    mask_zip2B   = boot_df_jR2R3_P301LB[:,0] < 0.7
    mask_pinch2B = boot_df_jR2R3_P301LB[:,1] < 0.8
    mask_2A = list(np.logical_and(mask_zip2A, mask_pinch2A))
    mask_2B = list(np.logical_and(mask_zip2B, mask_pinch2B))
    mask_jR2R3_P301L_intra = list(np.logical_or(mask_2A, mask_2B))
    new_jR2R3_P301L_hbonds = boot_jR2R3_P301L_hbonds[mask_jR2R3_P301L_intra]

    data_bootstrapped[k] = np.histogram(new_jR2R3_P301L_hbonds,bins=[5,np.inf],
                                        density=False)[0]/len(indices)

hbond_frac_confidence = np.zeros(3)
confidence = 0.90
sorted_frac = np.sort(data_bootstrapped)
il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
hbond_frac_confidence = low, mean, up
hbond_confidence_jR2R3_P301L_left = pd.DataFrame(np.reshape(hbond_frac_confidence,(1,3)),index=['jR2R3_P301L_left'],
                                columns=[f'{int(50-(confidence*100/2))}%','50%',f'{int(50+(confidence*100/2))}%'])

#%% NOW COMBINE AND PLOT
# combine the 4 hbond_confidence dataframes
hbond_confidence = pd.concat([hbond_confidence_jR2R3,hbond_confidence_jR2R3_P301L,hbond_confidence_jR2R3_left, hbond_confidence_jR2R3_P301L_left])

means = hbond_confidence.values[:,1]
errors = np.zeros((2,4))
for j,mean in enumerate(means):
    errors[0,j] = mean - hbond_confidence.values[j,0]
    errors[1,j] = hbond_confidence.values[j,2] - mean

barWidth=0.3
colors = ['#FF7F0E','#1F77B4','#FF7F0E','#1F77B4']
plt.bar([-0.3,0.0,0.7,1.0],means*100,yerr=errors*100,width=barWidth,color=colors,
        align='edge',edgecolor='black')

# output the means and errors into a csv file
error_down = errors[0]
error_up = errors[1]

df = pd.DataFrame({
    'Means': means,
    'Error Down': error_down,
    'Error Up': error_up
})

filename = "dimer_manyHbonds_meansErrors.csv"
df.to_csv(filename, index=False)
print(f"Data saved to {filename}")

# remove the x-ticks
plt.xticks([])

# have the y-ticks count by 5s
plt.yticks(np.arange(0, 25, 5))

# add manual x-tick labels
plt.text(0, -3, 'full dimer\nensemble', ha='center', va='bottom', fontsize=15)
plt.text(1, -4, 'pinched & clamped population\n(i.e. {K298-Q307 < 7$\AA$}\n& {V300-S305 < 8$\AA$})', ha='center', va='bottom', fontsize=12)
# plt.title('Reduced oligomerization when one monomer is closed',fontsize=15)
plt.ylabel('% probability of 5+ intermolecular\nbackbone hydrogen bonds',fontsize=12)

import matplotlib.patches as mpatches
# create custom colors and labels
color1 = '#FF7F0E'
color2 = '#1F77B4'
label1 = 'jR2R3'
label2 = 'jR2R3 P301L'

# create patch objects for the custom colors and labels
patch1 = mpatches.Patch(color=color1, label=label1)
patch2 = mpatches.Patch(color=color2, label=label2)

# create the legend
plt.legend(handles=[patch1, patch2],fontsize=18)
plt.tight_layout()

# save figure
plt.savefig('inter_hbond_confidence.png',dpi=300)

plt.show()
# %%
