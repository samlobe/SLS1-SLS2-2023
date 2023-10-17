#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import mdtraj as md
from itertools import combinations
import time

output_csv = True

#### LOAD TRAJECTORY ####
molecule = 'jR2R3'
traj = md.load(f'{molecule}.xtc',top=f'{molecule}.gro')
topology = traj.topology
print(traj)


#### GET PAIRWISE DISTANCES BETWEEN ALPHA-CARBONS ####
alpha_carbons = ([atom.index for atom in topology.atoms if atom.name == 'CA'])

atom_pairs = list(combinations(alpha_carbons,2))
pairwise_distances = md.geometry.compute_distances(traj[::1], atom_pairs)
pairwise_distances = np.around(pairwise_distances,3) # to thousandths of a nanometer

#### CREATE A LIST OF THE LABELS OF ALL YOUR DATA ####
labels = []

## First get a list of all the residues in your peptide
topology = traj.topology
res_names = [str(topology.residue(i)) for i in np.arange(topology.n_residues)]
res_names = [f'{name[:3]}{i+295}' for i,name in enumerate(res_names)]

## Then put all the pairs of residue names in your list of labels
res_pairs = list(combinations(np.arange(topology.n_residues),2))
for i, res in enumerate(res_names):
    for j in np.arange(i+1,len(res_names)):
        labels.append(f'Distance: {res_names[i]} and {res_names[j]}')
    
#### OUTPUT A CSV FILE IN YOUR WORKING DIRECTORY ####
if output_csv == True:
    start = time.time()
    df1 = pd.DataFrame(pairwise_distances, columns=labels)
    df1.to_csv(f'pair_distances_{molecule}.csv', index=False)
    end = time.time()
    print(f'Saving csv took {end-start:.2f} seconds')
