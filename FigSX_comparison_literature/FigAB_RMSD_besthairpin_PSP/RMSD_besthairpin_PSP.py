#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms

def get_RMSD(universe1,universe2,sel1,sel2):
    # tot_residues = len(universe2.residues)
    # weights = np.array([12]*tot_residues)
    R = rms.RMSD(universe1.select_atoms(sel1),  # coordinates to align
              universe2.select_atoms(sel2),  # reference coordinates
              weights=None,  # weights
              center=True,  # subtract the center of geometry
              superposition=True)  # superimpose coordinates
    R.run(verbose=True)
    results = R.results.rmsd[:,1:]
    results[:,1] = np.round(results[:,1]/10,4) #convert from Å to nm
    df = pd.DataFrame(results,columns=['time (ps)','RMSD to cluster 1 (nm)'])
    return df

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
def histo_line(data):
    bins = 50
    histo_height, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_middle = np.diff(bin_edges)/2
    bin_middle = bin_middle + bin_edges[:len(bin_middle)]
    return bin_middle, histo_height

ref1 = mda.Universe('best_hairpin.pdb')
ref2 = mda.Universe('PSP.gro')
u1 = mda.Universe('jR2R3.gro','jR2R3.xtc')
u2 = mda.Universe('jR2R3_P301L.gro','jR2R3_P301L.xtc')

# measure RMSD in each frame
jR2R3_RMSD_hairpin = get_RMSD(u1,ref1,'name CA','name CA')
jR2R3_P301L_RMSD_hairpin = get_RMSD(u2,ref1,'name CA','name CA')

jR2R3_RMSD_PSP = get_RMSD(u1,ref2,'name CA','name CA')
jR2R3_P301L_RMSD_PSP = get_RMSD(u2,ref2,'name CA','name CA')

# Save CSV files
jR2R3_RMSD_hairpin.to_csv('jR2R3_RMSD-to-hairpin.csv',index=False)
jR2R3_P301L_RMSD_hairpin.to_csv('jR2R3_P301L_RMSD-to-hairpin.csv',index=False)
jR2R3_RMSD_PSP.to_csv('jR2R3_RMSD-to-PSP.csv',index=False)
jR2R3_P301L_RMSD_PSP.to_csv('jR2R3_P301L_RMSD-to-PSP.csv',index=False)

### reload quickly
# jR2R3_RMSD_hairpin = pd.read_csv('jR2R3_RMSD-to-hairpin.csv')
# jR2R3_P301L_RMSD_hairpin = pd.read_csv('jR2R3_P301L_RMSD-to-hairpin.csv')
# jR2R3_RMSD_P301L = pd.read_csv('jR2R3_RMSD-to-PSP.csv')
# jR2R3_P301L_RMSD_PSP = pd.read_csv('jR2R3_P301L_RMSD-to-PSP.csv')
###

traj_dir = '.'
jR2R3_RMSD_hairpin = load(f'{traj_dir}/jR2R3_RMSD-to-hairpin.csv',50,10,10).iloc[:,1].to_numpy()
jR2R3_P301L_RMSD_hairpin = load(f'{traj_dir}/jR2R3_P301L_RMSD-to-hairpin.csv',50,10,10).iloc[:,1].to_numpy()
jR2R3_RMSD_PSP = load(f'{traj_dir}/jR2R3_RMSD-to-PSP.csv',50,10,10).iloc[:,1].to_numpy()
jR2R3_P301L_RMSD_PSP = load(f'{traj_dir}/jR2R3_P301L_RMSD-to-PSP.csv',50,10,10).iloc[:,1].to_numpy()

bin1,hist1 = histo_line(jR2R3_RMSD_hairpin)
bin2,hist2 = histo_line(jR2R3_P301L_RMSD_hairpin)

plt.plot(bin1,hist1,label='jR2R3',color='tab:orange')
plt.plot(bin2,hist2,label='jR2R3 P301L',color='tab:blue')

plt.xlabel('RMSD from hairpin (nm)',fontsize=15)
plt.ylabel('probability density',fontsize=15)
plt.ylim(bottom=0)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig('RMSD_hairpin.png',dpi=300)
plt.show()
#%%
bin1,hist1 = histo_line(jR2R3_RMSD_PSP)
bin2,hist2 = histo_line(jR2R3_P301L_RMSD_PSP)

plt.plot(bin1,hist1,label='jR2R3',color='tab:orange')
plt.plot(bin2,hist2,label='jR2R3 P301L',color='tab:blue')

plt.xlabel('RMSD from PSP (nm)',fontsize=15)
plt.ylabel('probability density',fontsize=15)
plt.ylim(bottom=0)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig('RMSD_PSP.png',dpi=300)

