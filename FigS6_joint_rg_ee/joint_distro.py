#%%
import mdtraj as md
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import islice

def array_cleaning(file_name, conv_time, simulation_sampling_freq, my_sampling_freq):
    # conv_time = convergence time (ns)
    # simulation_sampling_freq = the time step of the data in the file (ps)
    # my_sampling_freq = the time step you want (ps); probably the autocorrelation time
    skip = round(my_sampling_freq / simulation_sampling_freq)
    start = round(conv_time*1000/simulation_sampling_freq)
    clean_data = []
    with open(file_name, 'r') as f:
        for line in islice(f,start,None,skip):
            clean_data.append(line.split()[1])
    clean_data = np.asarray(clean_data)
    clean_data = clean_data.astype(float)
    return clean_data

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


molecules = ['jR2R3','jR2R3_P301L']
labels = ['jR2R3','jR2R3 P301L']
rgs = []
ees = []
pairdist_dir = '../pair_distances'
for molecule in molecules:
    rgs.append(array_cleaning(f'rg_{molecule}_alphaC.xvg',50,10,10))
    ees.append(load(f'{pairdist_dir}/pair_distances_{molecule}.csv',50,10,10)['Distance: ASP295 and VAL313'].values)

# combine ee and rg for each molecule into two dataframes
df1 = pd.DataFrame({'rg':rgs[0],'ee':ees[0]})
df2 = pd.DataFrame({'rg':rgs[1],'ee':ees[1]})
dfs = [df1,df2]

#%%
sns.set(font_scale = 1.5)
sns.set_style("whitegrid", {'axes.grid' : False})

def get_JointGrid(df,title,vmax=None):
    g = sns.JointGrid(data=df,x='ee',y=f'rg',xlim=[0,2],ylim=[0,5])
    cax = g.figure.add_axes([.90, .6, .02, .3])
    if isinstance(vmax,type(None)):
        g.plot_joint(sns.histplot, bins=50, binrange=(0,5), cmap='viridis',
                     cbar=True, cbar_ax=cax, stat='density')
    else:
        g.plot_joint(sns.histplot, bins=50, binrange=(0,5), cmap='viridis',
                     cbar=True, cbar_ax=cax, stat='density', vmax=vmax)
    
    g.plot_marginals(sns.histplot, kde=True, bins=50, stat='density')

    g.fig.get_axes()[1].set_title(title,fontsize=25)

    cbar = g.fig.colorbar(g.ax_joint.collections[0], cax=cax)
    cbar.set_label('Probability Density', rotation=90, labelpad=20)
    
    return g
    
g1 = get_JointGrid(df1,labels[0],vmax=3)
ax_joint1 = g1.fig.get_axes()[0]
ax_joint1.set_xlim(0,5)
ax_joint1.set_ylim(0.5,1.8)
ax_joint1.set_xlabel('End-to-end distance (nm)',fontsize=20)
ax_joint1.set_ylabel('Radius of gyration (nm)',fontsize=20)
plt.tight_layout()
plt.savefig('joint_distro_jR2R3.png',dpi=300)
plt.show()

g2 = get_JointGrid(df2,labels[1],vmax=3)
ax_joint2 = g2.fig.get_axes()[0]
ax_joint2.set_xlim(0,5)
ax_joint2.set_ylim(0.5,1.8)
ax_joint2.set_xlabel('End-to-end distance (nm)',fontsize=20)
ax_joint2.set_ylabel('Radius of gyration (nm)',fontsize=20)
plt.tight_layout()
plt.savefig('joint_distro_jR2R3_P301L.png',dpi=300)
plt.show()

# %%
