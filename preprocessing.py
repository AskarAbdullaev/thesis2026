import os
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torch_geometric.data import Data

pd.options.display.max_colwidth = 1000
pd.options.display.max_columns = 20
warnings.simplefilter("ignore")

from initial_tools import format_size


###############################################
# Part 4
#
# Sampling and Preprocessing
###############################################

STRUCT_MAP = {
 # Main Residues
 'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,

 # Modified Residues
 'PTR': 20, 'TPO': 21, 'SEP': 22, 'CSD': 23,  'TYS': 23, 'CGU': 23, 'CME': 23, 'LLP': 23,  # modified residues

 # Metals
 'MG': 24, 'ZN': 25, 'CA': 26, 'MN': 27, 'FE': 27, 'K': 27, 'NI': 27, 'HG': 27, 'CO': 27, # metals
 
 # Halogens
 'CL': 28, 'IOD': 28, 'F': 28, 'BR': 28, # halogens

 # DNA / RNA
 'DT': 29, 'DC': 29, 'DA': 29, 'DG': 29, 'T': 29, 'A': 29, 'C': 29, 'G': 29, 'U': 29, # nitrous bases

 # Phospho-Bases
 'UDP': 30, 'GDP': 30, 'CDP': 30, 'TDP': 30, 'ADP': 30,
 'UTP': 31, 'GTP': 31, 'CTP': 31, 'TTP': 31, 'ATP': 31,
 'UMP': 32, 'GMP': 32, 'CMP': 32, 'TMP': 32, 'AMP': 32,
 'UNP': 33, 'GNP': 33, 'CNP': 33, 'TNP': 33, 'ANP': 33,
 'NAD': 34, 'NDP': 34, 'NAP': 34, 'NAI': 34,
 
 # Popular cofactors
 'HEM': 35,
 'FAD': 36,
 'NAG': 37,
 'FMN': 38,
 'COA': 39,
 'SAH': 40, 
 'ACO': 41, 'CAA': 41, # CoA
 'CLA': 42, 'BCB': 42, 'BCL': 42, 'CHL': 42, # chlorophill
 'SAM': 43,

 # Carbohydrates
 'GLC': 44, 'BMA': 44, 'MAN': 44 # carbohydrates

}

STRUCT_MAP_UNKNOWN = 45

def split(df: pd.DataFrame,
          buckets: int = 12,
          dir: str = 'Files/Folds',
          attribute_column: str = 'html$AC',
          id_column: str = 'basic$scPDB ID',
          exist_ok: bool = True):
    """
    Deterministic split of data into N buckets, while assuring that
    values of attribute_column are unique in each bucket

    Args:
        df (pd.DataFrame): df of dataset
        buckets (int, optional): number of equal-sized buckets. Defaults to 12.
        dir (str, optional): place to store the buckets as text files with ids. Defaults to 'Files/Folds'.
        attribute_column (str, optional): attribute to group by. Defaults to 'html$AC'.
        id_column (str, optional): primary key column of df. Defaults to 'basic$scPDBID'.

    """
    
    assert isinstance(df, pd.DataFrame), f'df must be pd.DataFrame, not {type(df)}'
    assert isinstance(buckets, int) and buckets > 0, f'buckets must be positive integer, not {type(buckets)} ({buckets})'
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'
    assert isinstance(attribute_column, str), f'attribute_column must be str, not {type(attribute_column)}'
    assert isinstance(id_column, str), f'id_column must be str, not {type(id_column)}'
    assert attribute_column in df.columns, f'df has no column: {attribute_column}'
    assert id_column in df.columns, f'df has no column: {id_column}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'

    N = buckets
    buckets = {i: set() for i in range(N)}
    counts = {i: 0 for i in range(N)}

    print('Splitting the data')
    print(f'There are in total {len(df)} datapoints')
    print(f'The expected number per bucket: {len(df) // N} ({N})')

    def entropy(array: np.array):

        return - np.sum((array / np.sum(array)) * np.log(array / np.sum(array), where=(array > 0)))

    def try_adding(uniprot_id: str, to_bucket: int):

        counts_trial = counts.copy()
        n_added = df[attribute_column].value_counts()[uniprot_id]
        counts_trial[to_bucket] += n_added

        ent = entropy(np.array(list(counts_trial.values())))
        return ent

    def add(uniprot_id: str, to_bucket: int):

        entries_added = set(df.loc[df[attribute_column] == uniprot_id, id_column])
        buckets[to_bucket] |= entries_added
        counts[to_bucket] += len(entries_added)
        return
    
    if exist_ok and os.path.isdir(dir):
        print('Already exixts')
        for bucket in list(range(12)) + [101, 102, 103]:
            
            path = os.path.join(dir, f'{bucket}.txt')
            with open(path, 'r') as file:
                entries = file.read().split('\n')
            print(f'Bucket {bucket}: {len(entries)} entries')
        return

    for uniprot in tqdm(sorted(list(df[attribute_column].unique()))):

        entropies = np.array([try_adding(uniprot, j) for j in range(N)])
        best_bucket = np.argmax(entropies)
        add(uniprot, best_bucket)

    os.makedirs(dir, exist_ok=True)
    print(f'Buckets will be stored to {dir}')
    
    for bucket in buckets:
        path = os.path.join(dir, f'{bucket}.txt')
        with open(path,'w') as file:
            file.write('\n'.join(sorted(buckets[bucket])))
        print(f'Bucket {bucket}: {len(buckets[bucket])} entries')


    sub_df = df.loc[df[id_column].isin(buckets[10])].copy()
    N = 6
    buckets = {i: set() for i in range(N)}
    counts = {i: 0 for i in range(N)}

    print('Splitting the bucket N10')
    print(f'There are in total {len(sub_df)} datapoints')
    print(f'The expected number per bucket: {len(sub_df) // N} ({N})')

    for uniprot in tqdm(sorted(list(sub_df[attribute_column].unique()))):

        entropies = np.array([try_adding(uniprot, j) for j in range(N)])
        best_bucket = np.argmax(entropies)
        add(uniprot, best_bucket)

    print(f'Sub Buckets of N10 will be stored to {dir}')
    
    path = os.path.join(dir, '101.txt')
    with open(path,'w') as file:
        file.write('\n'.join(sorted(buckets[0] | buckets[1] | buckets[2] | buckets[3])))
    print(f'Bucket 10.1: {len(buckets[0] | buckets[1] | buckets[2] | buckets[3])} entries')

    path = os.path.join(dir, '102.txt')
    with open(path,'w') as file:
        file.write('\n'.join(sorted(buckets[4])))
    print(f'Bucket 10.2: {len(buckets[4])} entries')

    path = os.path.join(dir, '103.txt')
    with open(path,'w') as file:
        file.write('\n'.join(sorted(buckets[5])))
    print(f'Bucket 10.3: {len(buckets[5])} entries')

    return

RESIDUES = ['GLY', 'LEU', 'ILE', 'VAL', 'ALA', 'PRO', 'ASP', 'ASN', 'GLU', 'GLN',
            'ARG', 'LYS', 'HIS', 'CYS', 'MET', 'SER', 'THR', 'TYR', 'TRP', 'PHE']

GROUPS = ['ALIPHATIC', 'POLAR UNCHARGED', 'AROMATIC', 'POSITIVELY CHARGED', 'NEGATIVELY CHARGED']


def _sample_entry(scpdb_id: str,
                 dir: str = 'Files/scPDB',
                 dir_save: str = 'Files/Samples_Size_8',
                 cube_side: int = 8,
                 edge_radius: float = 4,
                 skip_existsing: bool = True) -> int:
    """
    A tool to sample a single entry of scPDB dataset: for each residue in the protein,
    take its alpha carbon coordinates and extract atoms that are inside

    Args:
        scpdb_id (str): entry id in scPDB dataset, e.g. '1a2o_1'
        dir (str, optional): directory to take entry files. Defaults to 'Files/scPDB'.
        dir_save (str, optional): directory to save arrays. Defaults to 'Files/Samples_Size_8'.
        cube_side (int, optional): the side of a subgraph. Defaults to 8.
        edge_radius (float, optional): the edge connection limit. Defaults to 4.
        skip_existsing (bool, optional): skip existing files. Defaults to True.

    Returns:
        int: size of saved files in bytes
    """
    
    # Check the input
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'
    assert isinstance(dir_save, str), f'dir_save must be str, not {type(dir_save)}'
    assert os.path.isdir(dir), f'no such directory: {dir}'
    assert isinstance(cube_side, int), f'cube_side must be int, not {type(cube_side)}'
    assert cube_side > 1, f'cube_side must be larger than 1, not {cube_side}'
    assert isinstance(scpdb_id, str), f'scpdb_id must be str, not {type(scpdb_id)}'
    assert isinstance(skip_existsing, bool), f'skip_existing must be bool, not {type(skip_existsing)}'

    # Check if csv's exist
    path_residues = os.path.join(dir, scpdb_id, 'residues.csv')
    path_atoms = os.path.join(dir, scpdb_id, 'atoms.csv')
    assert os.path.isfile(path_residues), f'residues.csv not found for: {scpdb_id}'
    assert os.path.isfile(path_atoms), f'atoms.csv not found for: {scpdb_id}'
    
    # For each residue in the protein: take its alpha carbon coordinates and extract atoms that are inside
    # a specified cubic domain inside it

    os.makedirs(dir_save, exist_ok=True)
    if skip_existsing:
        if os.path.isdir(os.path.join(dir_save, scpdb_id)):
            return 0
    os.makedirs(os.path.join(dir_save, scpdb_id), exist_ok=True)

    residues = pd.read_csv(os.path.join(dir, scpdb_id, 'residues.csv'))
    atoms = pd.read_csv(os.path.join(dir, scpdb_id, 'atoms.csv'))

    

    alpha_carbons_np = residues[['X ALPHA', 'Y ALPHA', 'Z ALPHA']].to_numpy()
    geo_center = alpha_carbons_np.mean(axis=0)
    longest_central_vector = np.max(np.linalg.norm(alpha_carbons_np - geo_center, axis=1))

    # Get distances between residues alpha-carbons
    distances = np.sum((alpha_carbons_np[:, np.newaxis, :] - alpha_carbons_np[np.newaxis, :, :]) ** 2, axis=-1)
    boolean_distances = (distances <= (cube_side / 2) ** 2)
    boolean_distances = boolean_distances.astype(bool)

    res_counter = Counter(residues['Substructure Name'].tolist())
    res_group_counter = Counter(residues['Residue Group'].tolist())

    x_span, y_span, z_span = atoms['X'].max() - atoms['X'].min(), atoms['Y'].max() - atoms['Y'].min(), atoms['Z'].max() - atoms['Z'].min()
    diagonal = np.sqrt(x_span ** 2 + y_span ** 2 + z_span ** 2)
    volume = int(x_span * y_span * z_span)
    res_coords = residues[['X ALPHA', 'Y ALPHA', 'Z ALPHA']].values
    geo_center = np.mean(res_coords, axis=0)

    # Calculate protein-level features
    protein_level_features = {
        'Number Of Residues': len(residues),
        'Number Of Chains': len(residues['Chain'].unique()),
        'X Span': x_span,
        'Y Span': y_span,
        'Z Span': z_span,
        'Volume': volume,
        **{f'Percent of {res}': round(res_counter[res] / len(residues), 5) for res in RESIDUES},
        'Total Charge': res_counter['ARG'] + res_counter['LYS'] - res_counter['ASP'] - res_counter['GLU'],
        'X Ratio': round(x_span / diagonal, 4),
        'Y Ratio': round(y_span / diagonal, 4),
        'Z Ratio': round(z_span / diagonal, 4),
        'Residues Density': round(len(residues) / volume, 5),
        'Atoms Density': round(len(atoms) / volume, 5),
        **{f'Percent of {g}': round(res_group_counter[g] / len(residues), 5) for g in GROUPS},
        'Diagonal': diagonal,
        'Binding Residues': int(residues['Binding'].sum())
    }

    # Save protein-level features
    with open(os.path.join(dir_save, scpdb_id, 'protein_features.npy'), 'wb') as f:
        np.save(f, np.array(list(protein_level_features.values())).astype(np.float32)[:-1])

    additional_features = []
    # Calculate residue-level geometric features and save them in the residues dataframe
    for res in range(boolean_distances.shape[1]):

        center = alpha_carbons_np[res]

        central_vector = center - geo_center
        central_vector_normalized_length = np.round(np.linalg.norm(central_vector) / longest_central_vector, 4)
        v1 = (alpha_carbons_np - center) / np.linalg.norm((alpha_carbons_np - center), axis=1)[:, np.newaxis]
        v1 = np.nan_to_num(v1, 0.0)
        v2 = central_vector / np.linalg.norm(central_vector) if np.linalg.norm(central_vector) > 0 else 0
        tilt = np.clip(np.dot(v1, v2), -1.0, 1.0)

        further_than_plane = np.round(np.sum(tilt > 0.00001) / len(residues), 5)
        cone = np.round(np.sum(tilt > 0.5) / len(residues), 5)
        cone_broad = np.round(np.sum(tilt > 0.25) / len(residues), 5)
        cone_sharp = np.round(np.sum(tilt > 0.75) / len(residues), 5)

        additional_features.append(np.array([central_vector_normalized_length, further_than_plane, cone, cone_broad, cone_sharp]))

    additional_features = np.vstack(additional_features)
    residues['Central Vector'], residues['Plane'], residues['Cone'], residues['Cone Broad'], residues['Cone Sharp']  = 0.0, 0.0, 0.0, 0.0, 0.0
    residues[['Central Vector', 'Plane', 'Cone', 'Cone Broad', 'Cone Sharp']] = additional_features

    size = 0
    # For each residue, save the features of close atoms and the edges between them
    for res in range(boolean_distances.shape[1]):
        
        res_name = residues.iloc[res]["Substructure Code"]
        
        if STRUCT_MAP.get(residues.iloc[res]["Substructure Name"], STRUCT_MAP_UNKNOWN) not in list(range(24)):
            continue
    
        res_chain = residues.iloc[res]["Chain"]
        label = residues.iloc[res]["Binding"]
        
        res_close = alpha_carbons_np[boolean_distances[:, res]]
        res_close_centered = res_close - alpha_carbons_np[res]
        
        res_close_df = residues.loc[boolean_distances[:, res]].copy()
        if len(res_close_df) < 3:
            continue
        res_close_df[['X', 'Y', 'Z']] = res_close_centered.astype(np.float16)
        res_close_df = res_close_df[['X', 'Y', 'Z', 'Substructure Name', 'Substructure Code', 'Central Vector', 'Plane', 'Cone', 'Cone Broad', 'Cone Sharp']]
        res_close_df['Substructure Name'] = res_close_df['Substructure Name'].map(lambda x: STRUCT_MAP.get(x, STRUCT_MAP_UNKNOWN))
        res_close_df['Central'] = False
        res_close_df.loc[res, 'Central'] = True

        path = os.path.join(dir_save, scpdb_id, f'resonly_{scpdb_id}_res_{res_name}_{res_chain}_{label}.npy')
        path_edges = os.path.join(dir_save, scpdb_id, f'resonly_{scpdb_id}_res_{res_name}_{res_chain}_{label}_edges.npy')

        res_features_to_save =res_close_df[['Substructure Name', 'X', 'Y', 'Z', 'Central', 'Central Vector', 'Plane', 'Cone', 'Cone Broad', 'Cone Sharp']].values
        # In the saved file:
        # 0: residue idx
        # 1, 2, 3: coordinates
        # 4: Central Mask
        # 5,6,7,8,9: Geo Features
        np.save(path, res_features_to_save.astype(np.float16))

        coords = res_close_df[['X', 'Y', 'Z']].values
        dists = np.sum((coords[np.newaxis, :, :] - coords[:, np.newaxis, :]) ** 2, axis=-1)
        mask = (dists <= (edge_radius ** 2)) & (dists > 0)

        edges = np.column_stack(np.nonzero(mask)).T.astype(np.int32)

        np.save(path_edges, edges)

        size += os.path.getsize(path)
        size += os.path.getsize(path_edges)
    
    return size


def entries_to_samples(dir: str = 'Files/scPDB',
                       dir_folds: str = 'Files/Folds',
                       dir_save: str = 'Files/Samples_Size_16',
                       cube_side: int = 16,
                       edge_radius: float = 3.0,
                       skip_existsing: bool = True) -> int:
    """
    Combined tool to convert entries of scPDB dataset into samples for training: for each residue in the protein,

    Args:
        dir (str, optional): directory to take entries from Defaults to 'Files/scPDB'.
        dir_folds (str, optional): directory with fold files. Defaults to 'Files/Folds'.
        dir_save (str, optional): directory to save arrays to. Defaults to 'Files/Samples_Size_16'.
        cube_side (int, optional): _description_. Defaults to 16.
        edge_radius (float, optional): edge cut-off. Defaults to 3.0.
        skip_existsing (bool, optional): skip already processed entries. Defaults to True.

    Returns:
        int: total size of saved files in bytes
    """

    # Check the input
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'
    assert isinstance(dir_save, str), f'dir_save must be str, not {type(dir_save)}'
    assert isinstance(dir_folds, str), f'dir_folds must be str, not {type(dir_folds)}'
    assert os.path.isdir(dir), f'no such directory: {dir}'
    assert os.path.isdir(dir_folds), f'no such directory: {dir_folds}'

    assert isinstance(cube_side, int), f'cube_side must be int, not {type(cube_side)}'
    assert cube_side > 1, f'cube_side must be larger than 1, not {cube_side}'
    assert isinstance(skip_existsing, bool), f'skip_existing must be bool, not {type(skip_existsing)}'

    print(f'Entries from {dir} will be converted to samples and saves into {dir_save}')

    all_entries = []
    total_size_all_folds = 0
    for fold_id in sorted(os.listdir(dir_folds)):

        if fold_id.split('.')[-1] != 'txt':
            continue
        all_entries = sorted(list(filter(lambda x: x, open(os.path.join(dir_folds, fold_id), 'r').read().split('\n'))))

        total_size = 0
        with tqdm(total=len(all_entries), desc=f'Computing Samples fold {fold_id}...') as pbar:
            for entry in all_entries:
                total_size += _sample_entry(entry,
                                            cube_side=cube_side, 
                                            dir_save=dir_save,
                                            dir=dir,
                                            edge_radius=edge_radius,
                                            skip_existsing=skip_existsing)
                pbar.set_description(f'Computing Samples fold {fold_id} ({format_size(total_size)}) ...')
                pbar.update(1)
        total_size_all_folds += total_size

    print(f'The total size of saved samples: {format_size(total_size_all_folds)}')
    
    return total_size_all_folds
