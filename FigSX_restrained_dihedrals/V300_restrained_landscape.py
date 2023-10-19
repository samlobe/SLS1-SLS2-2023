#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make a mask that counts frames when V300 (and L301) dihedrals resemble HP1's
columns = ['Angle phi: VAL300','Angle psi: VAL300','Angle phi: aa301','Angle psi: aa301']
dihedral_dir = '../dihedrals'
df_dihedrals = pd.read_csv(f'{dihedral_dir}/dihedrals_jR2R3_P301L.csv')[columns]

# make a mask for phi VAL300 < 0 and psi VAL300 > 1
V300_mask = np.logical_and(df_dihedrals['Angle phi: VAL300'].values < 0, df_dihedrals['Angle psi: VAL300'].values > 1.5)[5000:]

# make a mask for phi aa301 < 0 and greater than -1.8
aa301_mask = np.logical_and(df_dihedrals['Angle phi: aa301'].values < 0, df_dihedrals['Angle phi: aa301'].values > -3.14, df_dihedrals['Angle psi: aa301'].values > 1.5)[5000:]

mask = np.logical_and(V300_mask, aa301_mask)

# %%
def free_energy(a, b, T, y0, ymax, x0, xmax):
    free_energy, xedges, yedges = np.histogram2d(
        a, b, 30, [[y0, ymax], [x0, xmax]], density=True, weights=None)
    free_energy = np.log(np.flipud(free_energy)+.000001)
    free_energy = -(0.008314*T)*free_energy # for kJ/mol
    return free_energy, xedges, yedges

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

ensemble = 'jR2R3_P301L'
dist_dir = '../pair_distances'
df = load(f'{dist_dir}/pair_distances_{ensemble}.csv',50,10,10) #assuming it's converged at 50ns
#%%
distx = 'Distance: LYS298 and GLN307'
disty = 'Distance: VAL300 and SER305'
x = df[distx].to_numpy()[V300_mask]
y = df[disty].to_numpy()[V300_mask]

# %%
fig,ax = plt.subplots()
# cax = ax.set_axes([0.90, 0.25, 0.04, 0.5])


# CREATE A HEAT MAP OF THE FREE ENERGY OF THE CURRENT SIMULATION
dG, xedges, yedges = free_energy(y, x, 300, 0.25, 1.9, 0.15, 3.1)
dG = dG - np.min(dG)
im = plt.imshow(dG, interpolation='gaussian', extent=[
            yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap='jet',
            aspect='auto',vmax=14)


# FORMAT THE COLORBAR
cbar_ticks = np.arange(0,15,2)
cb = fig.colorbar(im)
cb.ax.get_yaxis().labelpad = 8
cb.ax.set_ylabel('kJ/mol',fontsize=13)

ax.set_title('jR2R3 P301L: restrained V300 dihedrals',fontsize=15)
ax.set_xlabel('Distance: K298 - Q307 (nm)',fontsize=14)
ax.set_ylabel('Distance: V300 - S305 (nm)',fontsize=14)

plt.tight_layout()
fig.savefig('V300_restrained_landscape.png',dpi=300,bbox_inches='tight')


plt.show()
# %%
