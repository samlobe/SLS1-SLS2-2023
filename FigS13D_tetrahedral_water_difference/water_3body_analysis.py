#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.interpolate import make_interp_spline, BSpline

nbins = 28
min_bin = 40
max_bin = 180
my_bins = np.arange(min_bin,max_bin+(max_bin-min_bin)/nbins,(max_bin-min_bin)/nbins)
bin_mids = np.diff(my_bins)/2
bin_mids = bin_mids + my_bins[:len(bin_mids)]

def histo_line(data):
    histo_height, bin_edges = np.histogram(data, bins=my_bins, density=True)
    bin_middle = np.diff(bin_edges)/2
    bin_middle = bin_middle + bin_edges[:len(bin_middle)]
    return bin_middle, histo_height

def read_angles(filepath):
    with open(filepath,'r') as f:
        angles = []
        for i,line in enumerate(f):
            if not line:
                print(f'Line {i} is empty.')
                angles.append([])
                continue
            else:
                angles.append([float(x) for x in line.split()])
                
    all_angles = np.array([item for sublist in angles for item in sublist])
    return all_angles

script_dir = os.path.dirname(__file__) # absolute dir the script is in
dirs = ['jR2R3','jR2R3_P301L']
seqs = ['DNIKHVPGGGSVQIVYKPV','DNIKHVLGGGSVQIVYKPV']

#%% PLOT THE BULK WATER 3-BODY ANGLE DISTRIBUTION
fig, ax = plt.subplots(figsize=(8,7))
script_dir = os.path.dirname(__file__) # absolute dir the script is in
bulk_file = f'{script_dir}/bulk_angles.txt'
all_angles = read_angles(bulk_file)
bins, bulk_distro = histo_line(all_angles)
plt.plot(bins,bulk_distro,label='bulk',lw=6,color='black')
plt.ylim(bottom=0,top=0.018)
plt.xlim(left=40,right=180)
plt.xlabel(r'Water 3-Body Angle ($\theta$)',fontsize=15)
plt.ylabel(r'$P(\theta)$',fontsize=15)

# INTEGRATE FROM 100-120 DEGREES TO FIND % OF TETRAHEDRAL WATER
tetrah = []
starti = np.argmin(np.abs(bins-102.5)) # 100-105 bin
endi   = np.argmin(np.abs(bins-117.5)) # 115-120 bin

# bulk waters
b_tetrahedral_ys = bulk_distro[starti:endi+1]
b_frac_tetrahedral = np.sum(b_tetrahedral_ys) * (bins[1]-bins[0]) # rectangular integration
b_frac_tetrahedral = np.around(b_frac_tetrahedral,5)
print(f'{b_frac_tetrahedral*100:.1f}% of bulk waters are tetrahedral')

#%% STRUCTURE 3-BODY DATA FOR 6 TEMPERATURES OF jR2R3 and jR2R3_P301L
# columns = groups (19 residues)
# index = bin_mids (42.5 to 177.5 degrees; each bin is 5 degrees from 40-180)

my_peptide = dirs[0]; my_seq = seqs[0]
df_data = []
group_names = []

for resid in tqdm(np.arange(295,313+1)):
    # Note: this is the 310K trajectory data
    backbone_file = f'{script_dir}/{my_peptide}/res{resid}_angles.txt'
    group_names.append(f'{my_seq[resid-295]}{resid}')
    all_angles = read_angles(backbone_file)
    _, histo = histo_line(all_angles)
    df_data.append(histo)
df_data = np.around(np.array(df_data).T,7)
jR2R3_data = pd.DataFrame(df_data,columns=group_names,index=bin_mids)
jR2R3_data.to_csv(f'jR2R3_3body.csv')

#%% same structure for jR2R3_P301L
my_peptide = dirs[1]; my_seq = seqs[1]
df_data = []
group_names = []
for resid in tqdm(np.arange(295,313+1)):
    backbone_file = f'{script_dir}/{my_peptide}/res{resid}_angles.txt'
    group_names.append(f'{my_seq[resid-295]}{resid}')
    all_angles = read_angles(backbone_file)
    _, histo = histo_line(all_angles)
    df_data.append(histo)
df_data = np.around(np.array(df_data).T,7)
jR2R3_P301L_data = pd.DataFrame(df_data,columns=group_names,index=bin_mids)
jR2R3_P301L_data.to_csv(f'jR2R3_P301L_3body.csv')

#%% IMPORT DATA FROM CSV FILES

jR2R3_data = pd.read_csv(f'jR2R3_3body.csv',index_col=0,header=0)
jR2R3_P301L_data = pd.read_csv(f'jR2R3_P301L_3body.csv',index_col=0,header=0)

#%% PLOT 3-BODY DISTRIBUTION[S] if you want

def plot_distro(peptide,my_distro='all',label='distro'):
    # peptide is 'jR2R3' or 'jR2R3_P301L'
    # distro is 'all' or the column name e.g. D295_backbone or I308_sidechain
    # label will default to the column name ('distro')
    df = pd.read_csv(f'{peptide}_3body.csv',index_col=0,header=0)
    if my_distro == 'all':
        for i,distro in enumerate(df.columns):
            if label == 'distro': plt.plot(df.index,df[distro],label=distro)
            else: plt.plot(df.index,df[distro],label=label)
    else:
        if label == 'distro': plt.plot(df.index,df[my_distro],label=my_distro)
        else: plt.plot(df.index,df[my_distro],label=label)

# plt.figure(figsize=(8,7))
plot_distro('jR2R3'); plt.title('jR2R3',fontsize=20)
# plot_distro('jR2R3_P301L')
# plot_distro('jR2R3','P301')
# plot_distro('jR2R3_P301L','L301')
plt.xlabel(r'3-body angle ($\theta$)',fontsize=15)
plt.ylabel('probability density',fontsize=15)
plt.ylim(bottom=0,top=0.018)
plt.legend(ncol=4,fontsize=7)

#%% Find the fraction of tetrahedral waters around each group

jR2R3_tet = list(jR2R3_data.loc[102.5:117.5].sum() * (bins[1]-bins[0])) # rectangular integration
jR2R3_P301L_tet = list(jR2R3_P301L_data.loc[102.5:117.5].sum() * (bins[1]-bins[0])) # rectangular integration
jR2R3_tet = np.around(np.array(jR2R3_tet),4)
jR2R3_tet = pd.Series(jR2R3_tet,index=jR2R3_data.columns)
jR2R3_tet = jR2R3_tet.rename(index = {'P301':'aa301'})
jR2R3_P301L_tet = np.around(np.array(jR2R3_P301L_tet),4)
jR2R3_P301L_tet = pd.Series(jR2R3_P301L_tet,index=jR2R3_P301L_data.columns)
jR2R3_P301L_tet = jR2R3_P301L_tet.rename(index = {'L301':'aa301'})

# combine the two dataframes
tet_frac = pd.concat([jR2R3_tet,jR2R3_P301L_tet],axis=1)
tet_frac.columns = ['jR2R3','jR2R3_P301L']

#%%
diff = jR2R3_P301L_tet.values - jR2R3_tet.values
diff = pd.Series(diff,index=jR2R3_tet.index)
plt.figure()
np.abs(diff*100).plot.bar()

plt.ylabel('difference in % tetrahedral waters\n(|jR2R3 P301L \u2212 jR2R3|)', fontsize=14)
plt.yticks(np.arange(0, 0.25, step=0.05),fontsize=14)
plt.xticks(fontsize=14)

# use consolas font for the xticks
for item in (plt.gca().get_xticklabels()):
    item.set_fontname('consolas')

# highlight the aa301 group
plt.gca().get_xticklabels()[6].set_color('red')

plt.tight_layout()
diff.to_csv('tetfrac_difference.csv',index='difference between jR2R3 and jR2R3_P301L') # positive means more tetrahedral in jR2R3_P301L

# save the figure
plt.savefig('tetfrac_difference.png',dpi=300)
plt.show()
