#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymbar import timeseries as ts
# note that the newer version of pymbar uses functions with underscores instead of uppercase letters between words
import mdtraj as md

ensemble = 'jR2R3'
sequence = 'DNIKHVPGGGSVQIVYKPV'
title = 'jR2R3'

# ensemble = 'jR2R3_P301L'
# sequence = 'DNIKHVLGGGSVQIVYKPV'
# title = 'jR2R3 P301L'

# ensemble = 'jR2R3_P301S'
# sequence = 'DNIKHVSGGGSVQIVYKPV'
# title = 'jR2R3 P301S'

# ensemble = 'jR2R3_P301V'
# sequence = 'DNIKHVVGGGSVQIVYKPV'
# title = 'jR2R3 P301V'

equilibration = 50 #ns
sim_sampling = 10 #ps
my_sampling = 10 #ps

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

traj_dir = '../trajectories'
traj = md.load(f'{traj_dir}/{ensemble}.xtc',top=f'{traj_dir}/{ensemble}.gro')
times = traj.time
n_residues = traj.n_residues

sec_struct = md.compute_dssp(traj,simplified=True)
# save sec_struct to csv
labels = [f'{res}{i+295}' for i,res in enumerate(sequence)]
sec_struct = pd.DataFrame(sec_struct,columns=labels)
sec_struct.to_csv(f'sec_struct_{ensemble}.csv',index=False)

sec_struct = load(f'sec_struct_{ensemble}.csv',equilibration,sim_sampling,my_sampling)
data = sec_struct.values != 'C'
labels = sec_struct.columns
sequence = [label[0] for label in labels]
sequence = ''.join(sequence)
n_residues = len(sequence)

#%%
bootstrap_num = 1000
data_bootstrapped = np.zeros((bootstrap_num,n_residues))
gs = np.zeros(n_residues)

for i, res_data in enumerate(data.T):
    print(f'Analyzing {labels[i]}...')

    try:
        C_n = ts.normalizedFluctuationCorrelationFunction(res_data) # compute the autocorrelation function
    except ts.ParameterError:
        print(f"ParameterError encountered for {labels[i]}. Skipping this iteration.")
        continue
    xs = np.arange(len(C_n)) * my_sampling / 1000 # convert to ns
    # Decide how much of the autocorrelation function to integrate based on the plot
    plt.plot(xs,C_n,lw=1,label=f'{labels[i]}')
    # plot a horizontal line at y=0
    plt.plot([0,xs[-1]],[0,0],'k--',lw=1)
    plt.xlim(left=0,right=100); plt.ylim(bottom=-.2,top=1)
    plt.show()
    integrate_up_to = 20 #ns
    integrate_index = np.argmin(np.abs(xs-integrate_up_to))
    autoC = np.sum(C_n[:integrate_index]) * my_sampling # integrate the autocorrelation function; `my_sampling` is the width of each bin in ps
    independent_frames = int(len(res_data) * my_sampling / (2*autoC)) # number of uncorrelated samples
    print(f'Number of uncorrelated samples: {independent_frames}\n')

    bootstrap_num = 1000

    for k in range(bootstrap_num):
        indices = np.random.choice(len(res_data),independent_frames)
        fraction_boot = np.sum(res_data[indices])/len(indices)
        data_bootstrapped[k,i] = fraction_boot

#%%

sec_struct_confidence = np.zeros((n_residues,3))
confidence = .90
for j,residue_array in enumerate(data_bootstrapped.T):
    sorted_frac = np.sort(residue_array)
    il, im, iu = round((1-confidence)/2*bootstrap_num),round(bootstrap_num/2),round((1-(1-confidence)/2)*bootstrap_num)
    low, mean, up = sorted_frac[il], sorted_frac[im], sorted_frac[iu]
    sec_struct_confidence[j,:] = low, mean, up
    
sec_struct_confidence = np.around(sec_struct_confidence,4)
sec_struct_confidence = pd.DataFrame(sec_struct_confidence,columns=['5%','50%','95%'],index=labels)
sec_struct_confidence.to_csv(f'sec_struct_frac_{ensemble}.csv') 

#%%
confidence = pd.read_csv(f'sec_struct_frac_{ensemble}.csv',index_col=0)

sec_struct_confidence = confidence.values
means = sec_struct_confidence[:,1]
errors = np.zeros((2,n_residues))
for j,mean in enumerate(means):
    errors[0,j] = mean - sec_struct_confidence[j,0]
    errors[1,j] = sec_struct_confidence[j,2] - mean

plt.bar(range(n_residues),means*100,yerr=errors*100,align='edge')
plt.xticks(np.arange(n_residues)+0.4,sequence)
plt.title(title,fontsize=25)
plt.ylabel(f'% probability of β-sheet or α-helix',fontsize=12)
plt.ylim(bottom=0,top=35)

# save figure
plt.savefig(f'beta_or_alpha_errors_{ensemble}.png',dpi=300,bbox_inches='tight')

plt.show()

#%%

