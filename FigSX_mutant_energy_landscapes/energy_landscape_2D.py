#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

titles = ['jR2R3','jR2R3_P301S','jR2R3_P301V','jR2R3 P301L']
molecules = ['jR2R3','jR2R3_P301S','jR2R3_P301V','jR2R3_P301L']
distances = ['Distance: LYS298 and GLN307','Distance: VAL300 and SER305']
dist_labels = ['Distance: K298 - Q307 (nm)','Distance: V300 - S305 (nm)']
dist_dir = '../pair_distances'

fig,axs = plt.subplots(1,4)
fig.set_figheight(3.5)
fig.set_figwidth(8)
cax = plt.axes([0.87, 0.18, 0.03, 0.708])

for i,molecule in enumerate(molecules):
    df = load(f'{dist_dir}/pair_distances_{molecule}.csv',50,10,10) #assuming it's converged at 50ns
    distx = distances[0]
    disty = distances[1]
    x = df[distx].to_numpy()
    y = df[disty].to_numpy()

    # CREATE A HEAT MAP OF THE FREE ENERGY OF THE CURRENT SIMULATION
    dG, xedges, yedges = free_energy(y, x, 300, 0.25, 1.9, 0.15, 3.1)
    dG = dG - np.min(dG)
    im = axs.flat[i].imshow(dG, interpolation='gaussian', extent=[
                yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap='jet',
                aspect='auto',vmax=14)

    axs[i].set_title(titles[i],fontsize=15)
    axs[i].set_ylabel(r'Distance: $\bf{V300}$ – $\bf{S305}$ (nm)',fontsize=14)
    axs[i].set_xlabel(r'Distance: $\bf{K298}$ – $\bf{Q307}$ (nm)',fontsize=14)

# FORMAT LABELS AND LAYOUT
for ax in axs.flat:
    ax.label_outer()
#fig.suptitle(res,fontsize=20)

# FORMAT THE COLORBAR
cbar_ticks = np.arange(0,16,2)
cb = fig.colorbar(im, cax=cax, ax=axs,ticks=cbar_ticks, format=('% .0f'),shrink=1)
cb.ax.get_yaxis().labelpad = 4
cb.ax.set_ylabel('kJ/mol',fontsize=13)

fig.tight_layout(rect=[0,0,.87,1])

# add arrows
for ax in axs:
    # Draw a white two-way arrow from (0.5,0.35) to (1.4,0.35) and write "unclamp" underneath it
    ax.annotate('',
            xy=(1.4, 0.37),  xycoords='data',
            xytext=(0.5, 0.37), textcoords='data',
            arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
    ax.text(0.95, 0.275, 'unclamp', fontsize=10, ha='center', color='white')
    # Draw a white vertical arrow from (0.35,0.6) to (0.35,1.3) and write "unpinch" to the left of it (rotated parallel to the arrow)
    ax.annotate('',
            xy=(0.35, 1.3),  xycoords='data',
            xytext=(0.35, 0.51), textcoords='data',
            arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
    ax.text(0.24, 0.9, 'unpinch', fontsize=10, va='center', ha='center', color='white', rotation='vertical')

# save figure
fig.savefig('mutant_energy_landscapes_2D.png',dpi=300,bbox_inches='tight')

plt.show()


# %%
