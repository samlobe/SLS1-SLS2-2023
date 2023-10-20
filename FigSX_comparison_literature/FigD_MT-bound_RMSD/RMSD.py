#%%

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda 
from MDAnalysis.analysis import rms
import pandas as pd
from tqdm import tqdm

def histo_line(data):
    bins = 50
    histo_height, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_middle = np.diff(bin_edges)/2
    bin_middle = bin_middle + bin_edges[:len(bin_middle)]
    return bin_middle, histo_height
def load(csv_file,ignore,sim_sampling,my_sampling):
    
    ''' Load a csv file efficiently. You can ignore the unequilibrated part and sample as frequently as you want.

    Parameters
    ----------
    csv_file : string
        The string of the csv file in your working directory.
    ignore : float (ns)
        How many ns of unequilibrated simulation should we ignore?
    sim_sampling : float (ps)
        What was the timestep of structures outputed in your simulation?
    my_sampling : float (ps)
        What timestep should we look at for this analysis?
        Pick a value close to the longest autocorrelation time to be efficient.
        Must be a multiple of sim_sampling.

    Returns
    -------
    Pandas data frame.
    '''
    ignore_index = int(ignore*1000/sim_sampling) # multiply by 1000 to convert to ps
    entries_to_skip = int(my_sampling / sim_sampling)
    df = pd.read_csv(csv_file, skiprows = lambda x: (x>0 and x<=ignore_index)
                     or (x%entries_to_skip and x>ignore_index))
    return df
def get_RMSD(universe1,universe2):
    R = rms.RMSD(universe1,  # coordinates to align
              universe2,  # reference coordinates
              select=f'(resid {300-294}-{304-294} or resid 300-304) and name CA',  # reference coordinates
              center=True,  # subtract the center of geometry
              superposition=True)  # superimpose coordinates
    R.run()
    results = R.results.rmsd[:,1:]
    results[:,1] = np.round(results[:,1],4)
    df = pd.DataFrame(results,columns=['time (ps)','RMSD to cluster 1 (nm)'])
    return df

MTbound = []
for i in np.arange(1,21):
    if i < 10: i = f'0{i}'
    MTbound.append(mda.Universe(f'tau_MT-bound_pdbs/tau_MT-bound_{i}.pdb')) # from PDB 2MZ7

u1 = mda.Universe('jR2R3.gro','jR2R3.xtc')
u2 = mda.Universe('jR2R3_P301L.gro','jR2R3_P301L.xtc')

#%% skip this cell if RMSD arrays are already calculated in csv file

jR2R3_RMSDs = np.zeros((len(u1.trajectory),len(MTbound)))
for i in tqdm(range(20)):
    jR2R3_RMSDs[:,i] = get_RMSD(u1,MTbound[i]).iloc[:,1]
np.savetxt('jR2R3_RMSDs.csv',jR2R3_RMSDs,delimiter=',')

jR2R3_P301L_RMSDs = np.zeros((len(u2.trajectory),len(MTbound)))
for i in tqdm(range(20)):
    jR2R3_P301L_RMSDs[:,i] = get_RMSD(u2,MTbound[i]).iloc[:,1]
np.savetxt('jR2R3_P301L_RMSDs.csv',jR2R3_P301L_RMSDs,delimiter=',')

#%%
jR2R3_RMSDs = np.loadtxt('jR2R3_RMSDs.csv',delimiter=',')
jR2R3_P301L_RMSDs = np.loadtxt('jR2R3_P301L_RMSDs.csv',delimiter=',')

jR2R3_RMSD = jR2R3_RMSDs.min(axis=1)
jR2R3_P301L_RMSD = jR2R3_P301L_RMSDs.min(axis=1)

bin1,hist1 = histo_line(jR2R3_RMSD)
bin2,hist2 = histo_line(jR2R3_P301L_RMSD)

plt.plot(bin1,hist1,label='jR2R3 no salt')
plt.plot(bin2,hist2,label='jR2R3_P301L no salt')

plt.xlabel(r'min RMSD C$\alpha$ ($\AA$)',fontsize=15)
plt.ylabel('probability density',fontsize=15)
plt.ylim(bottom=0)
plt.legend()

#%% PLOT CDF
plt.figure()

cdf1 = np.cumsum(hist1/sum(hist1))
cdf2 = np.cumsum(hist2/sum(hist2))

color1 = '#FF7F0E'
color2 = '#1F77B4'
plt.plot(bin1,cdf1,label='jR2R3',color='#FF7F0E')
plt.plot(bin2,cdf2,label='jR2R3 P301L',color='#1F77B4')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=15)
plt.xlabel(r'min RMSD C$\alpha$ ($\AA$)',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.ylim([0,1])
plt.xlim(left=0)
plt.tight_layout()
plt.savefig('RMSD_MT-bound_ensemble_CDF.png',dpi=300)
plt.show()

