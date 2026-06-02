import os
import json
import time

from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data

def read_fold(path: str):

    with open(path, 'r') as f:
        raw = f.read()

    clean = sorted(filter(lambda x: x, raw.split('\n')))
    return clean

def clean_name(s):

    if not isinstance(s, str):
        return s

    if 'virus' in s.lower():
        s = s.lower()
        s = s.split(' ')[:3]
        s = ' '.join(s).capitalize()

    else:
        s = s.lower()
        s = s.split(' ')[:2]
        s = ' '.join(s).capitalize()

    return s

class CustomDataSet(Dataset):

    entries_idx = sorted(os.listdir('Files/scPDB'))
    entry_to_idx = {name: i for i, name in enumerate(entries_idx)}

    def __init__(self,
                 use_folds: list[int],
                 reduction: int = 1,
                 dir_samples: str = 'Files/Residues_Only_25_8',
                 dir_folds: str = 'Files/Folds',
                 dir_main: str = 'Files/scPDB',
                 show_progress: bool = True):
        """
        Custom DataSet based on torch.utils.data.Dataset. 

        Returns:
            Data

        Args:
            use_folds (list[int | str]): indices of folds to use
            reduction (int, optional): reduction coefficient. Defaults to 1.
            dir_samples (str, optional): directory with samples. Defaults to 'Files/Samples_Size_16'.
            dir_folds (str, optional): directory with folds. Defaults to 'Files/Folds'.
            dir_main (str, optional): main directory. Defaults to 'Files/scPDB'.
            show_progress (bool, optional): show the progress of samples collection. Defaults to True.
        """
    
        # Check the input
        assert isinstance(use_folds, list), f'use_folds must be list, not {type(use_folds)}'
        assert isinstance(show_progress, bool), f'show_progress must be bool, not {type(show_progress)}'
        for f in use_folds:
            assert isinstance(f, int | str), f'fold indices must be int | str, not {type(f)}'
        assert isinstance(dir_samples, str), f'dir_samples must be str, not {type(dir_samples)}'
        assert isinstance(reduction, int), f'reduction must be int, not {type(reduction)}'
        assert reduction > 0, f'reduction must be positive, not {reduction}'
        assert isinstance(dir_folds, str), f'dir_folds must be str, not {type(dir_folds)}'
        assert isinstance(dir_main, str), f'dir_main must be str, not {type(dir_main)}'
        assert os.path.isdir(dir_samples), f'no such directory: {dir_samples}'
        

        # Main attricutes
        self.use_folds = use_folds
        self.dir_samples = dir_samples
        self.dir_main = dir_main
        self.dir_folds = dir_folds
        self.show_progress = show_progress
        self.reduction = reduction

        # Construction of index table that will map indices to datapoints
        self.fold_indices = {}

        counter = 0
        protein_counter = 0
        corrupted = 0
        corrupted_files = set()

        # First run to estimate the total runtime
        N = 0
        P = 0

        proteins_by_fold = {}
        for f_name in self.use_folds:
            fold_path = os.path.join(self.dir_folds, f'{f_name}.txt')
            fold_proteins = read_fold(fold_path)
            proteins_by_fold.update({f_name: set(fold_proteins)})
            for protein in fold_proteins[::self.reduction]:
                path_samples = os.path.join(self.dir_samples, protein)
                N += (len(os.listdir(path_samples)) // 2)
                P += 1
        
        N_res = N * 70
        N_edges = N * 300

        self.X = np.empty((N_res, 10), dtype=np.float32)
        self.dX = [0]
        self.dE = [0]
        self.y = np.empty((N, 1), dtype=np.float32)
        self.entry = np.empty((N, 1), dtype=np.int32)
        self.E = np.empty((2, N_edges), dtype=np.int64)
        self.P = np.empty((P, 38), dtype=np.float32)
        self.dP = []

        with tqdm(desc='Getting index table...', unit=' Files', disable=not self.show_progress, total=N) as pbar:

            # Iterate over folds
            for f_name in self.use_folds:

                self.fold_indices.update({f_name: []})

                # Get scPDB ids from folds directory:
                fold_proteins = proteins_by_fold[f_name]

                # Iterate over proteins
                for protein in sorted(fold_proteins)[::self.reduction]:

                    time.sleep(0.1)

                    path_samples = os.path.join(self.dir_samples, protein)
                    path_protein = os.path.join(path_samples, 'protein_features.npy')
                    
                    # Iterate over samples

                    sample_files = set(os.listdir(path_samples))

                    if 'protein_features.npy' not in sample_files:
                        continue

                    p = np.load(path_protein)
                    self.P[protein_counter] = p
                    
                    for f in sample_files:

                        if 'edges' in f or not 'resonly' in f:
                            continue

                        edge_path = f.replace('.npy', '_edges.npy')
                        if edge_path not in sample_files:
                            continue
                        
                        try:

                            x = np.load(os.path.join(path_samples, f))
                            y = int(f[-5])
                            e = np.load(os.path.join(path_samples, edge_path))
                            
                            # Store nodes with delimiter
                            last_index = self.dX[-1]
                            new_index = last_index + len(x)
                            if new_index >= len(self.X):
                                raise IndexError('self.X is depleted')
                            self.X[last_index:new_index] = x
                            self.dX.append(new_index)

                            # Store label
                            if counter >= len(self.y):
                                raise IndexError('self.y is depleted')
                            self.y[counter] = y
                            self.entry[counter] = self.entry_to_idx[protein]
                            
                            # Store edges with delimiter
                            last_index = self.dE[-1]
                            new_index = last_index + e.shape[1]
                            if new_index >= self.E.shape[1]:
                                raise IndexError('self.E is depleted')
                            if new_index > last_index:
                                self.E[:, last_index:new_index] = e
                            self.dE.append(new_index)

                            self.dP.append(protein_counter)

                            self.fold_indices[f_name].append(counter)
                            counter += 1
                        except Exception as e:

                            print(e)

                            corrupted += 1
                            corrupted_files.add(f)

                        x_occupancy = 100 * self.dX[-1] / len(self.X)
                        e_occupancy = 100 * self.dE[-1] / self.E.shape[1]
                        pbar.update(1) 

                        if counter % 1000 == 0:
                            pbar.set_description(f'Getting index table (X{x_occupancy:.2g}%/E{e_occupancy:.2g}%) (corrupted: {corrupted})...')
                    
                    protein_counter += 1

        self.X = self.X[:self.dX[-1]]
        self.X = np.nan_to_num(self.X, nan=0, posinf=0, neginf=0)
        self.E = self.E[:, :self.dE[-1]]
        self.E = np.nan_to_num(self.E, nan=0, posinf=0, neginf=0)
        self.y = self.y[:counter]
        self.P = np.nan_to_num(self.P, nan=0, posinf=0, neginf=0)

        self.P[:, 0] = np.log(1 + self.P[:, 0]) # Number of Residues
        self.P[:, 1] = np.log2(1 + self.P[:, 1]) # Number of Chains
        self.P[:, 2:5] = np.log2(1 + self.P[:, 2:5]) # Spans
        self.P[:, 5] = np.log10(1 + self.P[:, 5]) # Volume
        self.P[:, 26] = np.log(1 + self.P[:, 26]) # Charge
        self.P[:, -1] = np.log(1 + self.P[:, -1]) # Diagonal
        print(f'Corrupted_files: {corrupted_files}')


    def __str__(self):
        return f'DataSet, using fold(s): {self.use_folds}. Red: {self.reduction}. N Samples: {len(self)}.'
    
    def __repr__(self):
        return f'<{str(self)}>'

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index: int):

        start_x = self.dX[index]
        stop_x = self.dX[index+1]
        start_e = self.dE[index]
        stop_e = self.dE[index+1]

        x = self.X[start_x:stop_x]
        e = self.E[:, start_e:stop_e]
        y = self.y[index]
        entry = self.entry[index]
        p_index = self.dP[index]
        p = self.P[p_index]

        residue_idx = torch.from_numpy(x[:, 0]).to(torch.long).nan_to_num(nan=0, posinf=0, neginf=0)
        pos = torch.from_numpy(x[:, 1:4]).to(torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        edge_index = torch.from_numpy(e).to(torch.long).nan_to_num(nan=0, posinf=0, neginf=0)
        center_mask = torch.from_numpy(x[:, 4]).to(torch.bool)
        geo_features = torch.from_numpy(x[:, 5:]).to(torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        protein_features = torch.from_numpy(p).unsqueeze(0).to(torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return Data(
                    residue_idx=residue_idx,
                    pos=pos,
                    geo_features=geo_features,
                    protein_features=protein_features,
                    edge_index=edge_index,
                    center_mask=center_mask,
                    y=torch.from_numpy(y).to(torch.float32),
                    entry=torch.from_numpy(entry).to(torch.long),
                )


class Mapper:

    entries_idx = sorted(os.listdir('Files/scPDB'))

    def __init__(self, csv_file: pd.DataFrame):

        self._df = csv_file[['basic$scPDB ID', 'SCOPe', 'UniProt', 'EC', 'uniprot$Organism', 'html$Reign', 'mol2$Derived Ligand']]

        self._df.set_index(self._df['basic$scPDB ID'], inplace=True, drop=True)
        self._df = self._df.to_dict()

    def _clean_id(self, scpdb_id: str | int):

        if isinstance(scpdb_id, int):
            scpdb_id = self.entries_idx[scpdb_id]
        scpdb_id = scpdb_id.lower().strip()
        return scpdb_id

    def to_uniprot(self, scpdb_id: str | int):

        return self._df['UniProt'].get(self._clean_id(scpdb_id), None)
    
    def to_ligand(self, scpdb_id: str | int):

        return self._df['mol2$Derived Ligand'].get(self._clean_id(scpdb_id), None)
    
    def to_ec(self, scpdb_id: str | int):

        ec = self._df['EC'].get(self._clean_id(scpdb_id), None)
        if ec == '/':
            return None
        return ec
    
    def to_scope(self, scpdb_id: str | int):

        return self._df['SCOPe'].get(self._clean_id(scpdb_id), None)
    
    def to_organism(self, scpdb_id: str | int):

        org = self._df['uniprot$Organism'].get(self._clean_id(scpdb_id), None)
        return clean_name(org)
    
    def to_reign(self, scpdb_id: str | int):

        reign = self._df['html$Reign'].get(self._clean_id(scpdb_id), None)
        if not isinstance(reign, str):
            return None
        if 'bact' in reign.lower():
            return 'Bacteria'
        
        return reign
    