#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mdtraj as md
from scipy.interpolate import interp2d

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

### LOAD ALL THE SIMULATION DATA
df1 = load('dihedrals_jR2R3.csv',50,10,10)
df2 = load('dihedrals_jR2R3_P301L.csv',50,10,10)
simulation_data = [df1, df2]
simulations = ['jR2R3','jR2R3 P301L']

### GET RESIDUE NAMES FROM ANY HAIRPIN PDB FILE
sequence = 'DNIKHVXGGGSVQIVYKPV'
res_names = [f'{res}{id}' for res,id in zip(sequence,np.arange(295,313+1))]

π = np.pi # lol

# GET LANDSCAPES FOR RESIDUES YOU ARE INTERESTED IN
res = 'VAL300'
fig,axs = plt.subplots(2,1)
fig.set_figheight(8)
fig.set_figwidth(6)
plt.subplots_adjust(right=0.85)
cax = plt.axes([0.75, 0.125, 0.04, 0.755])

# LOOP THROUGH THE 2 SIMULATIONS
for j,df in enumerate(simulation_data):
    # GET SIMULATION DATA
    x = df[f'Angle phi: {res}'].to_numpy()
    y = df[f'Angle psi: {res}'].to_numpy()
    
    # CREATE A HEAT MAP OF THE FREE ENERGY OF THE CURRENT SIMULATION
    dG, xedges, yedges = free_energy(y, x, 300, -π, π, -π, π)
    dG = dG - np.min(dG)
    im = axs.flat[j].imshow(dG, interpolation='gaussian', extent=[
                    yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap='jet',
                    aspect='auto',vmax=20)
    axs.flat[j].set_aspect('equal')
    axs.flat[j].set_title(f'{simulations[j]}',fontsize=16)
    axs.flat[j].set_xlabel('$\Phi$',fontsize=14)
    axs.flat[j].set_ylabel('$\Psi$',fontsize=14)

    # set xticks and yticks from -3 to 3
    axs.flat[j].set_xticks(np.arange(-3,3.5,1))
    axs.flat[j].set_yticks(np.arange(-3,3.5,1))
    
    if j == 1:
        rect1 = patches.Rectangle((-3.11,1.5), 3.14, 1.60,
        linewidth=2,edgecolor='black',facecolor='none',alpha=0.5)
        axs.flat[j].add_patch(rect1)


# FORMAT LABELS AND LAYOUT
for ax in axs.flat:
    ax.label_outer()

# FORMAT THE COLORBAR
cbar_ticks = np.arange(0,25,5)
cb = fig.colorbar(im, cax=cax, ax=axs,ticks=cbar_ticks, format=('% .0f'), shrink=0.5)
cb.ax.get_yaxis().labelpad = 12
cb.ax.set_ylabel('kJ/mol',rotation=90,fontsize=14)

# add suptile
fig.suptitle(f'V300 dihedrals',fontsize=16)

plt.savefig('Ramachandran_V300.png',bbox_inches='tight')

plt.show()

