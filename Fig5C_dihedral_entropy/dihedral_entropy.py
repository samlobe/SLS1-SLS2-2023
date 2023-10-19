#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

def load(csv_file, ignore, sim_sampling, my_sampling):
    ignore_index = int(ignore * 1000 / sim_sampling)
    entries_to_skip = int(my_sampling / sim_sampling)
    return pd.read_csv(csv_file, skiprows=lambda x: (x > 0 and x <= ignore_index) or (x % entries_to_skip and x > ignore_index))

def entropy(values):
    values = np.array(values) + 0.000001
    return -0.008314 * np.sum(values * np.log(values))

def prob(values):
    return [values[values['Region'] == i].shape[0] / values.shape[0] for i in range(4)]

def assign(angles, res):
    regions = []
    for i in range(angles.shape[0]):
        phi = angles.at[i, f'Angle phi: {res}']
        psi = angles.at[i, f'Angle psi: {res}']
        
        if phi > 0:
            regions.append(0)
        elif phi < 0 and -120 < math.degrees(psi) < 50:
            regions.append(1)
        elif -100 < math.degrees(phi) < 0 and not (-120 < math.degrees(psi) < 50):
            regions.append(2)
        elif -180 < math.degrees(phi) <= -100 and not (-120 < math.degrees(psi) < 50):
            regions.append(3)
        else:
            regions.append(4)

    angles['Region'] = regions
    return angles

dihedral_dir = '../dihedrals'
df1 = load(f'{dihedral_dir}/dihedrals_jR2R3.csv', 50, 10, 10)
df2 = load(f'{dihedral_dir}/dihedrals_jR2R3_P301L.csv', 50, 10, 10)

sequence_1_letter = 'DNIKHVPGGGSVQIVYKPV'
res_names_1_letter = [f'{res}{i}' for i, res in enumerate(sequence_1_letter, 295)]
res_names_1_letter[6] = 'X301'

amino_acids = {'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE',
               'G':'GLY', 'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU',
               'M':'MET', 'N':'ASN', 'P':'PRO', 'Q':'GLN', 'R':'ARG',
               'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR'}

# Convert sequence to one-letter amino acid codes
sequence = [amino_acids[aa] for aa in sequence_1_letter]

res_names = [f'{res}{i}' for i, res in enumerate(sequence, 295)]
res_names[6] = 'aa301'
ires_names = res_names[1:-1]

# Create offset for bar positions
bar_width = 0.35
index = np.arange(len(ires_names))

# Handle plot drawing outside loop to avoid repeated labels
bars1 = []
bars2 = []

for res in tqdm(ires_names):
    for df, color, name in zip([df1, df2], ['#FF7F0E','#1F77B4'], ['jR2R3', 'jR2R3_P301L']):
        angles = df.loc[:, [f'Angle phi: {res}', f'Angle psi: {res}']]
        values = assign(angles, res)
        probs = prob(values)
        ent = entropy(probs) / 1e-3

        if name == 'jR2R3':
            bars1.append(ent)
        else:
            bars2.append(ent)

plt.bar(index, bars1, bar_width, color='#FF7F0E', label='jR2R3')
plt.bar(index + bar_width, bars2, bar_width, color='#1F77B4', label='jR2R3 P301L')

# Adjust xticks to the center of grouped bars
plt.ylabel('Backbone Dihedral Entropy ( J / mol·K )', fontsize=12)
plt.legend(fontsize=12)
plt.xticks(index + bar_width / 2, res_names_1_letter[1:-1], rotation=90,fontname='monospace')
# plt.xlabel('Residue', fontsize=15)
# plt.title('Residue Backbone Entropy', fontsize=15)
plt.tight_layout()
# save figure
plt.savefig('backbone_dihedral_entropy.png', dpi=300)
plt.show()


