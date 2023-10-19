#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

cutoff = 4.25
# interesting residues from compare_rings.py
interesting_residues = np.arange(300,302)
residues = 'DNIKHVXGGGSVQIVYKPV'
res_names = [f'{res}{resid+295}' for res,resid in zip(residues,range(19)) if resid+295 in interesting_residues]

color1 = '#FF7F0E'
color2 = '#1F77B4'

# independent frames from autocorrelation.py
jR2R3_indep_frames = [33551,34088]
jR2R3_P301L_indep_frames = [31380,33331]

jR2R3_dfs = {}
jR2R3_P301L_dfs = {}

for res_id in interesting_residues:
    jR2R3_dfs[str(res_id)] = pd.read_csv(f'jR2R3/res{res_id}_counting_rings.csv')['hexagons'].values
    jR2R3_P301L_dfs[str(res_id)] = pd.read_csv(f'jR2R3_P301L/res{res_id}_counting_rings.csv')['hexagons'].values

# normalize by number of waters
jR2R3_dfs_norm = {}
jR2R3_P301L_dfs_norm = {}
for res_id in interesting_residues:
    print(f'Looking at residue {res_id}')
    jR2R3_dfs_norm[str(res_id)] = pd.read_csv(f'jR2R3/res{res_id}_counting_rings.csv')['hexagons'].values / pd.read_csv(f'jR2R3/res{res_id}_counting_rings.csv')['num_waters'].values
    jR2R3_P301L_dfs_norm[str(res_id)] = pd.read_csv(f'jR2R3_P301L/res{res_id}_counting_rings.csv')['hexagons'].values / pd.read_csv(f'jR2R3_P301L/res{res_id}_counting_rings.csv')['num_waters'].values

#%%
bootstrap_num = 500
jR2R3_data_bootstrapped = np.zeros((bootstrap_num,len(interesting_residues)))
jR2R3_P301L_data_bootstrapped = np.zeros((bootstrap_num,len(interesting_residues)))

# bootstrap
for i,res_num in enumerate(interesting_residues):
    print(f'Looking at residue {res_num}...')
    data_jR2R3 = jR2R3_dfs_norm[str(res_num)]
    data_jR2R3_P301L = jR2R3_P301L_dfs_norm[str(res_num)]
    for k in tqdm(range(bootstrap_num)):
        indices_jR2R3 = np.random.choice(len(data_jR2R3),jR2R3_indep_frames[i])
        indices_jR2R3_P301L = np.random.choice(len(data_jR2R3_P301L),jR2R3_P301L_indep_frames[i])
        boot_jR2R3 = data_jR2R3[indices_jR2R3]
        boot_jR2R3_P301L = data_jR2R3_P301L[indices_jR2R3_P301L]
        jR2R3_data_bootstrapped[k,i] = np.mean(boot_jR2R3)
        jR2R3_P301L_data_bootstrapped[k,i] = np.mean(boot_jR2R3_P301L)

# sort each residue column and get confidence values
jR2R3_confidence = np.zeros((len(interesting_residues),3))
jR2R3_P301L_confidence = np.zeros((len(interesting_residues),3))

confidence = 0.90
for i in range(len(interesting_residues)):
    sorted_jR2R3 = np.sort(jR2R3_data_bootstrapped[:,i])
    sorted_jR2R3_P301L = np.sort(jR2R3_P301L_data_bootstrapped[:,i])
    il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
    low_jR2R3, mean_jR2R3, up_jR2R3 = sorted_jR2R3[il], sorted_jR2R3[im], sorted_jR2R3[iu]
    low_jR2R3_P301L, mean_jR2R3_P301L, up_jR2R3_P301L = sorted_jR2R3_P301L[il], sorted_jR2R3_P301L[im], sorted_jR2R3_P301L[iu]
    jR2R3_confidence[i,:] = low_jR2R3, mean_jR2R3, up_jR2R3
    jR2R3_P301L_confidence[i,:] = low_jR2R3_P301L, mean_jR2R3_P301L, up_jR2R3_P301L

jR2R3_confidence = pd.DataFrame(jR2R3_confidence,index=interesting_residues,
                                columns=[f'{int(50-(confidence*100/2))}%','50%',f'{int(50+(confidence*100/2))}%'])
jR2R3_P301L_confidence = pd.DataFrame(jR2R3_P301L_confidence,index=interesting_residues,
                                columns=[f'{int(50-(confidence*100/2))}%','50%',f'{int(50+(confidence*100/2))}%'])

# save as csv
jR2R3_confidence.to_csv(f'jR2R3_{cutoff}A_confidence_norm_hexagonal.csv')
jR2R3_P301L_confidence.to_csv(f'jR2R3_P301L_{cutoff}A_confidence_norm_hexagonal.csv')

# Plot the confidence intervals
jR2R3_confidence = pd.read_csv(f'jR2R3_{cutoff}A_confidence_norm_hexagonal.csv',index_col=0)
jR2R3_P301L_confidence = pd.read_csv(f'jR2R3_P301L_{cutoff}A_confidence_norm_hexagonal.csv',index_col=0)

jR2R3_means = jR2R3_confidence.values[:,1]
jR2R3_P301L_means = jR2R3_P301L_confidence.values[:,1]

jR2R3_errors = np.zeros((2,len(interesting_residues)))
jR2R3_P301L_errors = np.zeros((2,len(interesting_residues)))

for j,mean in enumerate(jR2R3_means):
    jR2R3_errors[0,j] = mean - jR2R3_confidence.values[j,0]
    jR2R3_errors[1,j] = jR2R3_confidence.values[j,2] - mean

for j,mean in enumerate(jR2R3_P301L_means):
    jR2R3_P301L_errors[0,j] = mean - jR2R3_P301L_confidence.values[j,0]
    jR2R3_P301L_errors[1,j] = jR2R3_P301L_confidence.values[j,2] - mean

plt.figure()
barWidth = 0.3
r1 = np.arange(len(interesting_residues))
r2 = [x + barWidth for x in r1]
plt.bar(r1,jR2R3_means,width=barWidth,color=color1,yerr=jR2R3_errors,
        align='center',edgecolor='black',label='jR2R3')
plt.bar(r2,jR2R3_P301L_means,width=barWidth,color=color2,yerr=jR2R3_P301L_errors,
        align='center',edgecolor='black',label='jR2R3_P301L')

res_names[1] = 'P/L301'
plt.xticks([r + barWidth/2 for r in range(len(interesting_residues))],res_names,fontsize=14)
plt.ylabel('Hexagonal Water Rings\n (normalized by # hydration waters)',fontsize=14)
plt.yticks(fontsize=12)
plt.ylim(top=0.014)
# plt.ylim(bottom=0.03)

import matplotlib.patches as mpatches
# create custom colors and labels
color1 = '#FF7F0E'
color2 = '#1F77B4'

label1 = 'jR2R3'
label2 = 'jR2R3_P301L'

# create patch objects for the custom colors and labels
patch1 = mpatches.Patch(color=color1, label=label1)
patch2 = mpatches.Patch(color=color2, label=label2)

# create the legend
plt.legend(handles=[patch1, patch2],fontsize=14)
plt.tight_layout()

# save figure
plt.savefig(f'hex_ring_{cutoff}A_confidence_norm.png',dpi=300)

plt.show()

#%% repeat for unnormalized data

bootstrap_num = 500
jR2R3_data_bootstrapped = np.zeros((bootstrap_num,len(interesting_residues)))
jR2R3_P301L_data_bootstrapped = np.zeros((bootstrap_num,len(interesting_residues)))

# bootstrap
for i,res_num in enumerate(interesting_residues):
    print(f'Looking at residue {res_num}...')
    data_jR2R3 = jR2R3_dfs[str(res_num)]
    data_jR2R3_P301L = jR2R3_P301L_dfs[str(res_num)]
    for k in tqdm(range(bootstrap_num)):
        indices_jR2R3 = np.random.choice(len(data_jR2R3),jR2R3_indep_frames[i])
        indices_jR2R3_P301L = np.random.choice(len(data_jR2R3_P301L),jR2R3_P301L_indep_frames[i])
        boot_jR2R3 = data_jR2R3[indices_jR2R3]
        boot_jR2R3_P301L = data_jR2R3_P301L[indices_jR2R3_P301L]
        jR2R3_data_bootstrapped[k,i] = np.mean(boot_jR2R3)
        jR2R3_P301L_data_bootstrapped[k,i] = np.mean(boot_jR2R3_P301L)

# sort each residue column and get confidence values
jR2R3_confidence = np.zeros((len(interesting_residues),3))
jR2R3_P301L_confidence = np.zeros((len(interesting_residues),3))

confidence = 0.90
for i in range(len(interesting_residues)):
    sorted_jR2R3 = np.sort(jR2R3_data_bootstrapped[:,i])
    sorted_jR2R3_P301L = np.sort(jR2R3_P301L_data_bootstrapped[:,i])
    il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
    low_jR2R3, mean_jR2R3, up_jR2R3 = sorted_jR2R3[il], sorted_jR2R3[im], sorted_jR2R3[iu]
    low_jR2R3_P301L, mean_jR2R3_P301L, up_jR2R3_P301L = sorted_jR2R3_P301L[il], sorted_jR2R3_P301L[im], sorted_jR2R3_P301L[iu]
    jR2R3_confidence[i,:] = low_jR2R3, mean_jR2R3, up_jR2R3
    jR2R3_P301L_confidence[i,:] = low_jR2R3_P301L, mean_jR2R3_P301L, up_jR2R3_P301L

jR2R3_confidence = pd.DataFrame(jR2R3_confidence,index=interesting_residues,
                                columns=[f'{int(50-(confidence*100/2))}%','50%',f'{int(50+(confidence*100/2))}%'])
jR2R3_P301L_confidence = pd.DataFrame(jR2R3_P301L_confidence,index=interesting_residues,
                                columns=[f'{int(50-(confidence*100/2))}%','50%',f'{int(50+(confidence*100/2))}%'])

# save as csv
jR2R3_confidence.to_csv('jR2R3_confidence_hexagonal.csv')
jR2R3_P301L_confidence.to_csv('jR2R3_P301L_confidence_hexagonal.csv')

# Plot the confidence intervals
jR2R3_confidence = pd.read_csv('jR2R3_confidence_hexagonal.csv',index_col=0)
jR2R3_P301L_confidence = pd.read_csv('jR2R3_P301L_confidence_hexagonal.csv',index_col=0)

jR2R3_means = jR2R3_confidence.values[:,1]
jR2R3_P301L_means = jR2R3_P301L_confidence.values[:,1]

jR2R3_errors = np.zeros((2,len(interesting_residues)))
jR2R3_P301L_errors = np.zeros((2,len(interesting_residues)))

for j,mean in enumerate(jR2R3_means):
    jR2R3_errors[0,j] = mean - jR2R3_confidence.values[j,0]
    jR2R3_errors[1,j] = jR2R3_confidence.values[j,2] - mean

for j,mean in enumerate(jR2R3_P301L_means):
    jR2R3_P301L_errors[0,j] = mean - jR2R3_P301L_confidence.values[j,0]
    jR2R3_P301L_errors[1,j] = jR2R3_P301L_confidence.values[j,2] - mean

plt.figure()
barWidth = 0.3
r1 = np.arange(len(interesting_residues))
r2 = [x + barWidth for x in r1]
plt.bar(r1,jR2R3_means*100,width=barWidth,color=color1,yerr=jR2R3_errors*100,
        align='center',edgecolor='black',label='jR2R3')
plt.bar(r2,jR2R3_P301L_means*100,width=barWidth,color=color2,yerr=jR2R3_P301L_errors*100,
        align='center',edgecolor='black',label='jR2R3_P301L')

res_names[1] = 'P/L301'
plt.xticks([r + barWidth/2 for r in range(len(interesting_residues))],res_names,fontsize=14)
plt.ylabel('% Probability of\nHexagonal Water Ring',fontsize=14)
plt.yticks(fontsize=14)

import matplotlib.patches as mpatches
# create custom colors and labels
label1 = 'jR2R3'
label2 = 'jR2R3_P301L'

# create patch objects for the custom colors and labels
patch1 = mpatches.Patch(color=color1, label=label1)
patch2 = mpatches.Patch(color=color2, label=label2)

# create the legend
plt.legend(handles=[patch1, patch2],fontsize=14)
plt.tight_layout()

# save figure
plt.savefig('hex_ring_confidence.png',dpi=300)

plt.show()
#%%
