# Usage from command line: `python count_rings.py ensemble_name res_num` where your structure file & trajectory file are labelled {ensemble_name}.gro & {ensemble_name}.xtc, and the res_num is the residue number.

import MDAnalysis as mda
from cycless.cycles import cycles_iter, centerOfMass
import pairlist
import numpy as np
import networkx as nx
from time import time
from tqdm import tqdm
import pandas as pd
import os
import sys

def water_HB_digraph(residue, u, cellmat):
    """Make a graph of hydrogen bonds of water molecules around a residue.
    Input: residue number (int), cell matrix (3x3 np.array of unit cell vectors)
    Output: directed graph of hydrogen bonds, fractional coordinates of water, number of waters
    """
    dg = nx.DiGraph()
    celli = np.linalg.inv(cellmat)

    my_residue = f'(resid {residue-294} and not name H*)'
    shell_waters = f'resname SOL and (around 6.0 {my_residue})'
    H = u.select_atoms(f'{shell_waters} and name H*').positions
    O = u.select_atoms(f'{shell_waters} and name O*').positions
    num_shell_waters = len(O)

    # In a fractional coordinate
    rH = H @ celli
    rO = O @ celli

    rH = rH.astype('float64')
    rO = rO.astype('float64')
    cellmat = cellmat.astype('float64')

    # O-H distance is closer than 2.45 AA
    # Matsumoto JCP 2007 https://doi.org/10.1063/1.2431168
    for i, j, d in pairlist.pairs_iter(rH, 2.45, cellmat, pos2=rO):
        # but distance is greater than 1 AA (i.e. O and H are not in the same
        # molecule)
        if 1 < d:
            # label of the molecule where Hydrogen i belongs.
            imol = i // 2
            # H to O vector
            # vec attribute is useful when you use cycless.dicycles.
            dg.add_edge(imol, j, vec=rO[j] - rH[i])
    return dg, rO, num_shell_waters

def main():
    ensemble = sys.argv[1]
    res_num = int(sys.argv[2])

    cycles_list = []
    num_waters_list = []
    n4, n5, n6 = [], [], []

    last_processed_frame = -1
    output_file = f'{ensemble}_res{res_num}_counting_rings.csv'

    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        last_processed_frame = df_existing.index[-1]
        # Load existing data into the lists
        n4 = df_existing['tetragons'].tolist()
        n5 = df_existing['pentagons'].tolist()
        n6 = df_existing['hexagons'].tolist()
        num_waters_list = df_existing['num_waters'].tolist()
        loaded_frames = len(num_waters_list)
        print(f'Loaded {loaded_frames} frames from {output_file}')
    else: loaded_frames = 0

    u = mda.Universe(f"{ensemble}.gro", f"{ensemble}.xtc")
    # cell dimension a,b,c,A,B,G
    # Note: length unit of MDAnalysis is AA, not nm.
    dimen = u.trajectory.ts.dimensions
    # cell matrix (might be transposed)
    cellmat = mda.lib.mdamath.triclinic_vectors(dimen)

    tik = time()
    for ts_index, ts in enumerate(tqdm(u.trajectory[last_processed_frame + 1:])):
        # make a graph of hydrogen bonds and fractional coordinate of its vertices
        dg, rO, num_waters = water_HB_digraph(res_num, u, cellmat) # around heavy atoms of residue 301 (P)
        num_waters_list.append(num_waters)

        # undirected graph
        g = nx.Graph(dg)
        # detect the tetragons, pentagons, and hexagons.
        cycles = [cycle for cycle in cycles_iter(
            g, maxsize=6) if len(cycle) > 3]
        cycles_list.append(cycles)

        # Save the dataframe after every 1000 iterations
        if (ts_index + 1) % 1000 == 0:
            for cycles in cycles_list:
                n4.append(sum(len(cycle) == 4 for cycle in cycles))
                n5.append(sum(len(cycle) == 5 for cycle in cycles))
                n6.append(sum(len(cycle) == 6 for cycle in cycles))
            df = pd.DataFrame({'tetragons': n4, 'pentagons': n5, 'hexagons': n6, 'num_waters': num_waters_list})
            df.to_csv(output_file, index=False)
            cycles_list = []
            tok = time()
            print(f'Finished processing frame {loaded_frames + ts_index + 1} for residue {res_num} after {(tok - tik)/60:.1f} minutes')

    # CONVERT CYCLES LIST TO n4, n5, n6

    labels = ['tetragons','pentagons','hexagons','num_waters']
    for cycles in cycles_list:
        n4.append(sum(len(cycle) == 4 for cycle in cycles))
        n5.append(sum(len(cycle) == 5 for cycle in cycles))
        n6.append(sum(len(cycle) == 6 for cycle in cycles))

    df = pd.DataFrame({'tetragons': n4, 'pentagons': n5, 'hexagons': n6, 'num_waters': num_waters_list})

    # save dataframe to csv
    df.to_csv(output_file, index=False)

    print(f'\n\nCompleted counting water rings for {len(df)} frames of trajectory for residue {res_num}')

if __name__ == '__main__':
    main()

