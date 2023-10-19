#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import mdtraj as md
import time

output_csv = True # Decide if you want to output the csv

traj_dir = '../trajectories'
molecules = ['jR2R3','jR2R3_P301L']

for molecule in molecules:
    print(f'\nMeasuring dihedrals in {molecule}.xtc')
    traj = md.load(f'{traj_dir}/{molecule}.xtc',top=f'{traj_dir}/{molecule}.gro')
    topology = traj.topology
    print(traj)
    
    print('Measuring phi angles...')
    _,phi = md.compute_phi(traj)
    print('Measuring psi angles...')
    _,psi = md.compute_psi(traj)
    print('Measuring omega angles...')
    _,omega = md.compute_omega(traj)
    
    angles = np.hstack((phi,psi,omega))
    angles = np.around(angles,3) # to thousandths of a radian
    
    #%%
    #### CREATE A LIST OF THE LABELS OF ALL YOUR DATA ####
    labels = []
    
    ## First get a list of all the residues in your peptide
    res_names_bad = [str(topology.residue(i)) for i in np.arange(topology.n_residues)]
    res_names = [f'{res[:3]}{i+295}' for i,res in enumerate(res_names_bad)]
    res_names[6] = 'aa301'
    
    for res in np.arange(phi.shape[1]):
        labels.append(f'Angle phi: {res_names[res+1]}')
    for res in np.arange(psi.shape[1]):
        labels.append(f'Angle psi: {res_names[res]}')
    for res in np.arange(omega.shape[1]):
        labels.append(f'Angle omega: {res_names[res]}-{res_names[res+1]}')
        
    #%% OUPUT CSV
    df = pd.DataFrame(angles, columns=labels)
    df.to_csv(f'dihedrals_{molecule}.csv', index=False)
    print(f'Outputted dihedrals_{molecule}.csv') 
