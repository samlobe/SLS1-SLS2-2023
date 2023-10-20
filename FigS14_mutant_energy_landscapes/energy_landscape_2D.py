#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md
from scipy.interpolate import interp2d

dist_dir = '../pair_distances'
molecule = 'jR2R3' # HIE protonation state
# molecule = 'jR2R3_P301L'
title = 'jR2R3'
# title = 'jR2R3 P301L'

def free_energy(a, b, T, y0, ymax, x0, xmax, weights=None):
    free_energy, xedges, yedges = np.histogram2d(
        a, b, 30, [[y0, ymax], [x0, xmax]], density=True, weights=weights)
    free_energy = np.log(np.flipud(free_energy)+.000001)
    free_energy = -(0.008314*T)*free_energy # for kJ/mol
    # use 0.008314 for kJ/mol
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

df = load(f'{dist_dir}/pair_distances_{molecule}.csv',50,10,10) # converged by 50ns

distx = 'Distance: LYS298 and GLN307'
disty = 'Distance: VAL300 and SER305'

x = df[distx].to_numpy()
y = df[disty].to_numpy()

fig,ax = plt.subplots()

# CREATE A HEAT MAP OF THE FREE ENERGY OF THE CURRENT SIMULATION
dG, xedges, yedges = free_energy(y, x, 300, 0.25, 1.9, 0.15, 3.1)
dG = dG - np.min(dG)
im = plt.imshow(dG, interpolation='gaussian', extent=[
            yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap='jet',
            aspect='auto',vmax=12)

# FORMAT THE COLORBAR
cbar_ticks = np.arange(0,15,2)
cb = fig.colorbar(im)
cb.ax.get_yaxis().labelpad = 12
cb.ax.set_ylabel('kJ/mol',rotation=90,fontsize=13)

# Draw a white two-way arrow from (0.5,0.35) to (1.4,0.35) and write "unclamp" underneath it
ax.annotate('',
            xy=(1.4, 0.37),  xycoords='data',
            xytext=(0.5, 0.37), textcoords='data',
            arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
ax.text(0.95, 0.275, 'unclamp', fontsize=14, ha='center', color='white')

# Draw a white vertical arrow from (0.35,0.6) to (0.35,1.3) and write "unpinch" to the left of it (rotated parallel to the arrow)
ax.annotate('',
            xy=(0.35, 1.3),  xycoords='data',
            xytext=(0.35, 0.51), textcoords='data',
            arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
ax.text(0.24, 0.9, 'unpinch', fontsize=14, va='center', ha='center', color='white', rotation='vertical')

ax.set_title(title,fontsize=20)
ax.set_xlabel('Distance: K298 - Q307 (nm)',fontsize=14)
ax.set_ylabel('Distance: V300 - S305 (nm)',fontsize=14)

ax.set_title(title,fontsize=20)
ax.set_xlabel('Distance: K298 - Q307 (nm)',fontsize=14)
ax.set_ylabel('Distance: V300 - S305 (nm)',fontsize=14)

plt.tight_layout()
fig.savefig('P301_energy_landscape_2D.png',dpi=300,bbox_inches='tight')
# fig.savefig('P301L_energy_landscap_2D.png',dpi=300,bbox_inches='tight')

plt.show()

