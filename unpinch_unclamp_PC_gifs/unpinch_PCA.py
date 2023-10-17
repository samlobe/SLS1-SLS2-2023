#%%
import numpy as np
import pandas as pd
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import pca, align
import warnings
import mdtraj as md
warnings.filterwarnings('ignore')
from time import time
import matplotlib.pyplot as plt
import mdtraj as md
import os

#%%
traj_dir = '../trajectories'
u0 = mda.Universe(f'{traj_dir}/jR2R3_P301L.gro',f'{traj_dir}/jR2R3_P301L.xtc')
dist_dir = '../pair_distances'
df = pd.read_csv(f'{dist_dir}/pair_distances_jR2R3_P301L.csv',usecols=['Distance: LYS298 and GLN307','Distance: VAL300 and SER305'])
x = df['Distance: LYS298 and GLN307'].values
y = df['Distance: VAL300 and SER305'].values

xmask = np.logical_and(x > 0.53, x < 0.58)
ymask = np.logical_and(y > 0.5, y < 1.4)
mask = np.logical_and(xmask,ymask)

#%%
selection = u0.select_atoms('all')
with mda.Writer('unpinch.xtc', selection.n_atoms) as W:
    for ts in u0.trajectory[mask]:
        W.write(selection)
#%% I used  Gromacs to cluster the trajectory ^

#%% Now I want to align the trajectory to cluster1.pdb
tik = time()
u = mda.Universe(f'{traj_dir}/jR2R3_P301L.gro','unpinch.xtc')
pdb = mda.Universe('cluster_zone1.pdb')
aligner = align.AlignTraj(u, pdb, select='backbone',
                          in_memory=True).run()
tok = time(); print(f'Done aligning to cluster_zone1.pdb in {tok-tik:.2f} seconds') # 13 seconds

#%%
tik = time()
pdb_bb = pdb.select_atoms('backbone')
pc = pca.PCA(u,select='backbone',align=True, mean_atoms=pdb_bb,
             n_components=None).run()
tok = time(); print(f'Done with PCA in {tok-tik:.2f} seconds') # 

#%%
backbone = u.select_atoms('backbone')
n_bb = len(backbone)
print('There are {} backbone atoms in the analysis'.format(n_bb))
print(pc.p_components.shape)

#%%
#for i in range(7):
#    print(f"Cumulated variance: {pc.cumulated_variance[i]:.3f}")
plt.plot(np.arange(1,11),pc.cumulated_variance[:10])
plt.xlabel('Principal component')
plt.ylabel('Cumulative variance'); plt.ylim(bottom=0,top=1)

#%%
transformed = pc.transform(backbone, n_components=3)
transformed.shape

df = pd.DataFrame(transformed,columns=['PC{}'.format(i+1) for i in range(3)])
df['Time (ps)'] = df.index * u.trajectory.dt

df.head()




#%% PLOT AND ANIMATE THE PRINCIPAL COMPONENT
which_pc = 7 # 
my_pc = pc.p_components[:, which_pc-1].reshape(n_bb, 3)
cluster_backbone = pdb.select_atoms('backbone')
cluster_positions = cluster_backbone.positions

def create_pc_values(start, end):
    if start < 0:
        down_array = np.linspace(0, start, int(abs(start/0.25))+1)
        mid_array = np.linspace(start+0.25, end-0.25, int(abs((end-start)/0.25))-1)
        up_array = np.linspace(end, 0, int(abs(end/0.25))+1)
        pc_values = np.concatenate((down_array, mid_array, up_array))
    else:
        up_array = np.linspace(0, start, int(abs(start/0.25))+1)
        mid_array = np.linspace(start+0.25, end-0.25, int(abs((end-start)/0.25))-1)
        down_array = np.linspace(end, 0, int(abs(end/0.25))+1)
        pc_values = np.concatenate((up_array, mid_array, down_array))
    return pc_values

pc_values = create_pc_values(-9,4)

protein_positions = cluster_backbone.positions
# make a directory 'unpinch_pdb_snapshots' to store snapshots
if not os.path.exists('unpinch_pdb_snapshots'):
    os.makedirs('unpinch_pdb_snapshots')

for i,pc_value in enumerate(pc_values):
    cluster_backbone.positions = protein_positions + pc_value*my_pc
    # save the snapshot of the protein with the corresponding PC1 value
    cluster_backbone.write(f"unpinch_pdb_snapshots/pc{which_pc}_snapshot{i}.pdb".format(pc_value))

pdb_files = [f"unpinch_pdb_snapshots/pc{which_pc}_snapshot{snapshot}.pdb".format(pc_value) for snapshot in np.arange(i+1)]

trajs = [md.load(pdb) for pdb in pdb_files]
stitched_traj = md.join(trajs)
stitched_traj.save(f'unpinch_pc{which_pc}.pdb')

#%% draw the pc onto the energy landscape
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

titles = ['jR2R3','jR2R3 P301L']
molecules = ['jR2R3','jR2R3_P301L']
distances = ['Distance: LYS298 and GLN307','Distance: VAL300 and SER305']
dist_labels = ['Distance: K298 - Q307 (nm)','Distance: V300 - S305 (nm)']

fig,axs = plt.subplots(1,2)
fig.set_facecolor('white')
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
    axs[i].set_xlabel(dist_labels[0],fontsize=14)
    axs[i].set_ylabel(dist_labels[1],fontsize=14)

# FORMAT LABELS AND LAYOUT
for ax in axs.flat:
    ax.label_outer()
#fig.suptitle(res,fontsize=20)

# FORMAT THE COLORBAR
cbar_ticks = np.arange(0,15,2)
cb = fig.colorbar(im, cax=cax, ax=axs,ticks=cbar_ticks, format=('% .0f'),shrink=1)
cb.ax.get_yaxis().labelpad = 4
cb.ax.set_ylabel('kJ/mol',fontsize=13)

fig.tight_layout(rect=[0,0,.87,1])

# Draw a white two-way arrow from (0.5,0.35) to (1.4,0.35) and write "unclamp" underneath it
axs[0].annotate('',
            xy=(1.4, 0.37),  xycoords='data',
            xytext=(0.5, 0.37), textcoords='data',
            arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
axs[0].text(0.95, 0.275, 'unclamp', fontsize=10, ha='center', color='white')
axs[1].annotate('',
            xy=(1.4, 0.37),  xycoords='data',
            xytext=(0.5, 0.37), textcoords='data',
            arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
axs[1].text(0.95, 0.275, 'unclamp', fontsize=10, ha='center', color='white')

# Draw a white vertical arrow from (0.35,0.6) to (0.35,1.3) and write "unpinch" to the left of it (rotated parallel to the arrow)
axs[0].annotate('',
            xy=(0.35, 1.3),  xycoords='data',
            xytext=(0.35, 0.51), textcoords='data',
            arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
axs[0].text(0.24, 0.9, 'unpinch', fontsize=10, va='center', ha='center', color='white', rotation='vertical')
axs[1].annotate('',
            xy=(0.35, 1.3),  xycoords='data',
            xytext=(0.35, 0.51), textcoords='data',
            arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
axs[1].text(0.24, 0.9, 'unpinch', fontsize=10, va='center', ha='center', color='white', rotation='vertical')

# Now measure the two distances in pc{which_pc}.pdb and plot it over
from MDAnalysis.analysis import distances
# open the trajectory pc{which_pc}.pdb
u_new = mda.Universe(f'unpinch_pc{which_pc}.pdb')
# select the first pair of atoms (x-axis)
pair1a = u_new.select_atoms(f'resid {298-294} and name CA'); pair1b = u_new.select_atoms(f'resid {307-294} and name CA')
pair2a = u_new.select_atoms(f'resid {300-294} and name CA'); pair2b = u_new.select_atoms(f'resid {305-294} and name CA')
dist1s = []; dist2s = []
for ts in u_new.trajectory:
    _,_,dist1 = distances.dist(pair1a, pair1b)
    _,_,dist2 = distances.dist(pair2a, pair2b)
    dist1s.append(dist1[0]/10); dist2s.append(dist2[0]/10)

# dist1s = [np.around(dist,2) for dist in dist1s]
# dist2s = [np.around(dist,2) for dist in dist2s]
# print('clamp dist:'); print(dist1s)
# print('pinch dist:'); print(dist2s)
# plot dist1s and dist2s over the energy landscape
# ax.plot(dist1s,dist2s,marker='o', markerfacecolor='none', markeredgecolor='black', markersize=10)


# Try to animate it
from matplotlib.animation import FuncAnimation
dot1 = axs[1].scatter(dist1s[0], dist2s[0], s=30, color='white')
dot2 = axs[0].scatter(dist1s[0], dist2s[0], s=30,facecolor='None' ,edgecolor='white',lw=1)
def update(num):
    x,y = dist1s[num], dist2s[num]
    dot1.set_offsets([x, y])
    dot2.set_offsets([x, y])
    return dot1,dot2

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(len(u_new.trajectory)), 
                    interval=33.3333, blit=True)
plt.show()
ani.save("unpinch_just_landscape.gif",dpi=200)


# %%
