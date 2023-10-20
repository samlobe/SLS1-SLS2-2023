#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# equilibration time (from correlation_time.py)
equil_jR2R3 = 110 #ns
equil_jR2R3_P301L = 80  #ns

# load alpha carbon pairwise distances data
dist_dir = '../pair_distances/dimers'
columns_jR2R3 = ['Distance: LYS298 and GLN307','Distance: VAL300 and SER305']
columns_jR2R3_P301L = ['Distance: LYS298 and GLN307','Distance: VAL300 and SER305']
df_jR2R3A = pd.read_csv(f'{dist_dir}/jR2R3_monA.csv',usecols=columns_jR2R3).sort_index()[equil_jR2R3*100:]
df_jR2R3B = pd.read_csv(f'{dist_dir}/jR2R3_monB.csv',usecols=columns_jR2R3).sort_index()[equil_jR2R3*100:]
df_jR2R3_P301LA = pd.read_csv(f'{dist_dir}/jR2R3_P301L_monA.csv',usecols=columns_jR2R3_P301L).sort_index()[equil_jR2R3_P301L*100:]
df_jR2R3_P301LB = pd.read_csv(f'{dist_dir}/jR2R3_P301L_monB.csv',usecols=columns_jR2R3_P301L).sort_index()[equil_jR2R3_P301L*100:]

# load the number of intermolecular hydrogen bond per frame (created with  `gmx hbond ...` with mainchain+H selected for each monomer)
hbond_dir = '../Fig4D_dimer_interHbonds'
data_jR2R3_hbonds = np.loadtxt(f'{hbond_dir}/jR2R3_hbnum.xvg',comments=['@','#'])[equil_jR2R3*100:,1]
data_jR2R3_P301L_hbonds = np.loadtxt(f'{hbond_dir}/jR2R3_P301L_hbnum.xvg',comments=['@','#'])[equil_jR2R3_P301L*100:,1]

# num independent
indep_jR2R3 = 1274
indep_jR2R3_P301L = 1139

#%% BOOTSTRAP FOR jR2R3

bootstrap_num = 500
data_bootstrapped = np.zeros((bootstrap_num,10))
bins = np.arange(11)-0.5

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR2R3_hbonds),indep_jR2R3)
    boot_jR2R3_hbonds = data_jR2R3_hbonds[indices]
    boot_df_jR2R3A = df_jR2R3A.values[indices]
    boot_df_jR2R3B = df_jR2R3B.values[indices]
    data_bootstrapped[k] = np.histogram(boot_jR2R3_hbonds,bins=bins,density=True)[0]

hbond_frac_confidence = np.zeros((10,3))
confidence = 0.90

for j,hbond_num_array in enumerate(data_bootstrapped.T):
    sorted_frac = np.sort(hbond_num_array)
    il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
    low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
    hbond_frac_confidence[j,:] = low, mean, up

hbond_confidence = pd.DataFrame(hbond_frac_confidence,columns=[f'{50-(confidence*100/2)}%','50%',f'{50+(confidence*100/2)}%'])

means = hbond_confidence.values[:,1]
errors = np.zeros((2,10))
for j,mean in enumerate(means):
    errors[0,j] = mean - hbond_confidence.values[j,0]
    errors[1,j] = hbond_confidence.values[j,2] - mean

plt.bar(range(10),means*100,yerr=errors*100,align='edge')
plt.xticks(np.arange(10)+0.4,np.arange(10))
plt.title('jR2R3',fontsize=30)
plt.ylabel(r'% of ensemble',fontsize=15)
plt.xlabel('# of intermolecular\nbackbone hydrogen bonds',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(bottom=0,top=25)
plt.xlim(left=0.8,right=10)
plt.tight_layout()
plt.savefig('jR2R3_interHbond_detailed.png')
plt.show()

# %% NOW FOR jR2R3_P301L
bootstrap_num = 500
data_bootstrapped = np.zeros((bootstrap_num,10))
bins = np.arange(11)-0.5

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR2R3_P301L_hbonds),indep_jR2R3_P301L)
    boot_jR2R3_P301L_hbonds = data_jR2R3_P301L_hbonds[indices]
    boot_df_jR2R3_P301LA = df_jR2R3_P301LA.values[indices]
    boot_df_jR2R3_P301LB = df_jR2R3_P301LB.values[indices]
    data_bootstrapped[k] = np.histogram(boot_jR2R3_P301L_hbonds,bins=bins,density=True)[0]

hbond_frac_confidence = np.zeros((10,3))
confidence = 0.90

for j,hbond_num_array in enumerate(data_bootstrapped.T):
    sorted_frac = np.sort(hbond_num_array)
    il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
    low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
    hbond_frac_confidence[j,:] = low, mean, up

hbond_confidence = pd.DataFrame(hbond_frac_confidence,columns=[f'{50-(confidence*100/2)}%','50%',f'{50+(confidence*100/2)}%'])

means = hbond_confidence.values[:,1]
errors = np.zeros((2,10))
for j,mean in enumerate(means):
    errors[0,j] = mean - hbond_confidence.values[j,0]
    errors[1,j] = hbond_confidence.values[j,2] - mean

plt.bar(range(10),means*100,yerr=errors*100,align='edge')
plt.xticks(np.arange(10)+0.4,np.arange(10))
plt.title('jR2R3 P301L',fontsize=30)
plt.ylabel(r'% of ensemble',fontsize=15)
plt.xlabel('# of intermolecular\nbackbone hydrogen bonds',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(bottom=0,top=25)
plt.xlim(left=0.8,right=10)
plt.tight_layout()
plt.savefig('jR2R3_P301L_interHbond_detailed.png')
plt.show()
# %% NOW DO jR2R3 AGAIN BUT FILTERING FOR AT LEAST ONE MONOMER
# IN THE LOWER LEFT OF THE ENERGY LANDSCAPE (r298-307 < 0.7A) AND (r300-305 < 0.8A)

bootstrap_num = 500
data_bootstrapped = np.zeros((bootstrap_num,10))
bins = np.arange(11)-0.5

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

    data_bootstrapped[k] = np.histogram(new_jR2R3_hbonds,bins=bins,density=True)[0]

hbond_frac_confidence = np.zeros((10,3))
confidence = 0.90

for j,hbond_num_array in enumerate(data_bootstrapped.T):
    sorted_frac = np.sort(hbond_num_array)
    il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
    low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
    hbond_frac_confidence[j,:] = low, mean, up

hbond_confidence = pd.DataFrame(hbond_frac_confidence,columns=[f'{50-(confidence*100/2)}%','50%',f'{50+(confidence*100/2)}%'])

means = hbond_confidence.values[:,1]
errors = np.zeros((2,10))
for j,mean in enumerate(means):
    errors[0,j] = mean - hbond_confidence.values[j,0]
    errors[1,j] = hbond_confidence.values[j,2] - mean

plt.bar(range(10),means*100,yerr=errors*100,align='edge')
plt.xticks(np.arange(10)+0.4,np.arange(10))
plt.title('jR2R3:\nwhen one monomer is pinched/clamped',fontsize=15)
plt.xlabel('# of intermolecular\nbackbone hydrogen bonds',fontsize=15)
plt.ylabel(r'% of ensemble',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(bottom=0,top=25)
plt.xlim(left=0.8,right=10)
plt.tight_layout()
plt.savefig('jR2R3_closed_interHbond_detailed.png')
plt.show()

# %% NOW DO jR2R3_P301L AGAIN BUT FILTERING FOR AT LEAST ONE MONOMER
# IN THE LOWER LEFT OF THE ENERGY LANDSCAPE (r298-307 < 0.7A / r300-305 < 0.8A)

bootstrap_num = 500
data_bootstrapped = np.zeros((bootstrap_num,10))
bins = np.arange(11)-0.5

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

    data_bootstrapped[k] = np.histogram(new_jR2R3_P301L_hbonds,bins=bins,density=True)[0]

hbond_frac_confidence = np.zeros((10,3))
confidence = 0.90

for j,hbond_num_array in enumerate(data_bootstrapped.T):
    sorted_frac = np.sort(hbond_num_array)
    il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
    low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
    hbond_frac_confidence[j,:] = low, mean, up

hbond_confidence = pd.DataFrame(hbond_frac_confidence,columns=[f'{50-(confidence*100/2)}%','50%',f'{50+(confidence*100/2)}%'])

means = hbond_confidence.values[:,1]
errors = np.zeros((2,10))
for j,mean in enumerate(means):
    errors[0,j] = mean - hbond_confidence.values[j,0]
    errors[1,j] = hbond_confidence.values[j,2] - mean

plt.bar(range(10),means*100,yerr=errors*100,align='edge')
plt.xticks(np.arange(10)+0.4,np.arange(10))
plt.title('jR2R3 P301L:\nwhen one monomer is pinched/clamped',fontsize=15)
plt.ylabel(r'% of ensemble',fontsize=15)
plt.xlabel('# of intermolecular\nbackbone hydrogen bonds',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(bottom=0,top=25)
plt.xlim(left=0.8,right=10)
plt.tight_layout()
plt.savefig('jR2R3_P301L_closed_interHbond_detailed.png')
plt.show()
# %%