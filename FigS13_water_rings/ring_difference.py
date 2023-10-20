#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# looking at 4.25A cutoff data (analyzing 310K trajectory)
peptide_labels = ['jR2R3','jR2R3 P301L']
peptides = ['jR2R3','jR2R3_P301L']
seqs = ['DNIKHVPGGGSVQIVYKPV','DNIKHVLGGGSVQIVYKPV']
res_ids = np.arange(295,314)
columns = ['tetragons', 'pentagons', 'hexagons']

#%%
# Process the data for each peptide and each cutoff
data_dfs = {}
data_dfs_norm = {}
for peptide in peptides:
    peptide_data_dfs = {}
    peptide_data_dfs_norm = {}
    res_data = []
    res_data_norm = []
    for res_id in res_ids:
        data = pd.read_csv(f'{peptide}/res{res_id}_counting_rings.csv').mean()
        res_data.append(data[:-1].values)
        res_data_norm.append(data[:-1].values / data['num_waters'])
    df = pd.DataFrame(np.array(res_data),columns=columns,index=res_ids)
    df_norm = pd.DataFrame(np.array(res_data_norm),columns=columns,index=res_ids)
    data_dfs[peptide] = df
    data_dfs_norm[peptide] = df_norm

#%%
# measure the difference in pentagonal and hexagonal rings (normalized by waters)
cutoff = '4.25A'
plt.figure()

jR2R3_P301L_hexagons = []
jR2R3_hexagons = []
jR2R3_P301L_pentagons = []
jR2R3_pentagons = []

for i, res_id in enumerate(res_ids):
    jR2R3_P301L_hexagons.append(data_dfs_norm['jR2R3_P301L'].loc[res_id, 'hexagons'])
    jR2R3_hexagons.append(data_dfs_norm['jR2R3'].loc[res_id, 'hexagons'])
    jR2R3_P301L_pentagons.append(data_dfs_norm['jR2R3_P301L'].loc[res_id, 'pentagons'])
    jR2R3_pentagons.append(data_dfs_norm['jR2R3'].loc[res_id, 'pentagons'])

# calculate the difference
hexagon_diff = [a - b for a, b in zip(jR2R3_P301L_hexagons, jR2R3_hexagons)]
pentagon_diff = [a - b for a, b in zip(jR2R3_P301L_pentagons, jR2R3_pentagons)]

plt.plot(res_ids, hexagon_diff, 'h-', markersize=10, color='purple',label='hexagons')
plt.plot(res_ids, pentagon_diff, 'p-', markersize=10, color='black',label='pentagons')

# plot the horizontal line at 0
plt.axhline(y=0, color='black', linestyle='--')

residues = 'DNIKHVXGGGSVQIVYKPV'
plt.xticks(res_ids, residues, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(top=0.004)

plt.xlabel('Residue ID', fontsize=15)
plt.ylabel('jR2R3 P301L \u2212 jR2R3 Ring Difference\n(normalized by # hydration waters)', fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
# plt.title(f'Difference in Normalized Penaton/Hexagon Count at 310K for {cutoff} cutoff', fontsize=20)
plt.tight_layout()
plt.savefig(f'water_ring_difference_cutoff_425A.png', dpi=300)

plt.show()
#%%