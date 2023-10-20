#%%
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from tqdm import tqdm

traj_dir = '../../trajectories'
u1 = mda.Universe(f'{traj_dir}/jR2R3.gro',f'{traj_dir}/jR2R3.xtc')
u2 = mda.Universe(f'{traj_dir}/jR2R3_P301L.gro',f'{traj_dir}/jR2R3_P301L.xtc')

def get_dist(u):
    atom1 = u.select_atoms(f'resid {300-294} and name O')
    atom2 = u.select_atoms(f'resid {303-294} and name N')
    print(list(atom1.resnames))
    print(list(atom2.resnames))

    conv = 50000 ; timestep = u.trajectory[1].time #ps
    frame1 = int(conv/timestep)
    
    dist_array = []
    for ts in tqdm(u.trajectory[frame1:]):
        dist_array.append(float(distances.dist(atom1, atom2)[2]))
    dist_array = np.array(np.around(dist_array,3))
    
    return dist_array

dist1 = get_dist(u1)
dist2 = get_dist(u2)

#%% PLOT PROBABILITY DENSITY

def histo_line(data):
    bins = 50
    histo_height, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_middle = np.diff(bin_edges)/2
    bin_middle = bin_middle + bin_edges[:len(bin_middle)]
    return bin_middle, histo_height

plt.figure()
bin1,hist1 = histo_line(dist1)
bin2,hist2 = histo_line(dist2)

plt.plot(bin1,hist1,label='jR2R3')
plt.plot(bin2,hist2,label='jR2R3 P301L')

plt.xlabel('distance ($\AA$)',fontsize=15)
plt.ylabel('probability density',fontsize=15)
plt.ylim(bottom=0)
plt.legend(fontsize=15)

#%% PLOT CDF
plt.figure()

cdf1 = np.cumsum(hist1/sum(hist1))
cdf2 = np.cumsum(hist2/sum(hist2))

plt.plot(bin1,cdf1,label='jR2R3',color='tab:orange')
plt.plot(bin2,cdf2,label='jR2R3 P301L',color='tab:blue')

plt.legend(fontsize=15)
plt.xlabel('distance ($\AA$): O(V300)-N(G303)',fontsize=15)
plt.ylabel('CDF',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim([0,1])
plt.xlim([2,9])
plt.tight_layout()
plt.savefig('O300-N303.png',dpi=300)
plt.show()
