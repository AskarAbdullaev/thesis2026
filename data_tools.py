import os
import json
import warnings

import pandas as pd
import numpy as np
import regex as re

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path as PathLib

from scpdb import SCPDB

pd.options.display.max_colwidth = 1000
pd.options.display.max_columns = 20
warnings.simplefilter("ignore")

###############################################
# Aggregation and Analysis
###############################################

def aggregate_dictoinaries(dir: str = 'Files/scPDB',
                           df_version: int = 4,
                           df_dir: str = 'Files',
                           start_from: int = 0,
                           from_compiled: bool = False,
                           save_residues: bool = True,
                           save_atoms: bool = True,
                           save_fasta: bool = True,
                           save_json: bool = True) -> pd.DataFrame:
    """
    Iterates over folders from "dir", which look like PDB codes.
    Creates SCPDB instances, extracts properties as dictionaries.
    By request, can also save to each folder:
        - properties as "compiled.json"
        - dataframe with residues and their properties as "residues.csv"
        - FASTA sequences of each chain in a whole (FASTA.txt) and as a mask for binding site (FASTAsite.txt)

    Args:
        dir (str, optional): directory to search for PDB folders. Defaults to 'Files/scPDB'.
        df_version (int, optional): version to attach to the result. Defaults to 4.
        df_dir (str, optional): directory to save the database to_. Defaults to 'Files'.
        start_from (int, optional): starting index of the sorted folders. Defaults to 0.
        from_compiled (bool, optional): just read compiled jsons. Defaults to False.
        save_residues (bool, optional): save residues DF as csv in each folder. Defaults to True.
        save_fasta (bool, optional): save FASTAs as txt in each folder. Defaults to True.
        save_json (bool, optional): save extracted properties as json in each folder. Defaults to True.

    Returns:
        pd.DataFrame: compiled database of properties
    """

    assert isinstance(dir, str), f'dir must be a str, not {type(dir)}'
    assert os.path.isdir(dir), f'no such directory: {dir}'
    assert isinstance(df_version, int | str), f'df_version must be int | str, not {type(df_version)}'
    assert isinstance(df_dir, str), f'df_dir must be a str, not {type(df_dir)}'
    assert os.path.isdir(df_dir), f'no such directory: {df_dir}'
    assert isinstance(start_from, int), f'start_from must be int, not {type(start_from)}'
    assert isinstance(save_residues, bool), f'save_residues must be a bool, not {type(save_residues)}'
    assert isinstance(save_fasta, bool), f'save_fasta must be a bool, not {type(save_fasta)}'
    assert isinstance(save_json, bool), f'save_json must be a bool, not {type(save_json)}'
    assert isinstance(from_compiled, bool), f'from_compiled must be a bool, not {type(from_compiled)}'

    table = []
    folders = list(filter(lambda x: '_' in x and len(x) <= 7, os.listdir(dir)))

    # Iterate through folders

    with tqdm(desc='Gathering Properties...', total=len(folders) - start_from) as pbar:
        for folder in sorted(folders)[start_from:]:

            pbar.set_description(f'Gathering Properties ({folder})...')

            if from_compiled:
                path = os.path.join(dir, folder, 'compiled.json')
                if os.path.isfile(path):
                    with open(path, 'r') as js:
                        table.append(dict(json.load(js)))
                pbar.update(1)
                continue

            # Create an instance
            t = SCPDB(folder)

            # Get properties
            d = t.dict
            table.append(d)

            # Save if required
            if save_json:
                with open(os.path.join(dir, folder, 'compiled.json'), 'w') as f:
                    json.dump({k: str(v) for k, v in d.items()}, f)
        
            if save_residues:
                t.mol2.protein.Save_Residues()

            if save_atoms:
                t.mol2.protein.Save_Atoms()

            if save_fasta:
                t.mol2.Save_FASTA()
                t.mol2.Save_FASTAsite()

            pbar.update(1)
            

    # Create a DF
    df = pd.DataFrame(table)
    df.to_csv(os.path.join(df_dir, f'database_v{df_version}.csv'), sep='@')

    return df



# Specification of columns grouped by type of information
COLUMN_GROUPS =lambda df: {

    # Basic IDs: need to count unique numbers and groups of duplicates
    'Basic IDs': {'columns': ['basic$scPDB ID', 'basic$PDB ID', 'html$Name', 'UniProt', 'EC',  'SCOPe',  'SCOP1', 'html$Uniprot ID', 'html$AC', 'html$TaxID', 'html$EC', 'html$EC1','html$EC12',
                              'html$EC123', 'html$SCOPe Dominant Chain', 'html$SCOPe1 Dominant Chain', 'html$SCOPe12 Dominant Chain',
                              'html$SCOPe123 Dominant Chain', 'rcsb$rcsb_entry_container_identifiers$pubmed_id', "basic$UniRef50"],
                  'kwargs': {'moments': False, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

    # Statistics for the presence of files
    'Completeness': {'columns': ['basic$Protein Mol2', 'basic$Site Mol2', 'basic$Cavity Mol2', 'basic$Interaction Mol2', 'basic$IPF txt',
                                 'basic$HTML JSON', 'basic$HTML txt', 'basic$RCSB JSON', 'basic$RCSB identifiers', 'basic$Minimum Requirements',],
                     'kwargs': {'moments': False, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

    # Statistics for the Number of of Files
    'NFiles': {'columns': ['basic$RCSB Entities', 'basic$RCSB Assemblies', 'basic$RCSB Poly Entities', 'basic$RCSB Non Poly',
                          'basic$RCSB Branched', 'rcsb$rcsb_entry_info$entity_count', 'rcsb$rcsb_entry_info$assembly_count', 
                          'rcsb$rcsb_entry_info$branched_entity_count', 'rcsb$rcsb_entry_info$cis_peptide_count',
                          'rcsb$rcsb_entry_info$nonpolymer_entity_count', 'rcsb$rcsb_entry_info$polymer_entity_count',
                          'rcsb$rcsb_entry_info$polymer_entity_count_dna', 'rcsb$rcsb_entry_info$polymer_entity_count_rna',
                          'rcsb$rcsb_entry_info$polymer_entity_count_nucleic_acid', 'rcsb$rcsb_entry_info$polymer_entity_count_nucleic_acid_hybrid',
                          'rcsb$rcsb_entry_info$polymer_entity_count_protein', 'rcsb$Number of Entities', 'rcsb$Number of Polymers',
                          'rcsb$Number of Non Polymers', 'rcsb$Number of Proteins', 'rcsb$Number of Branched', 'rcsb$Number of DNAs',
                          'rcsb$Number of RNAs', 'rcsb$Number of NAs', 'rcsb$Number of NA Hybrids',],
                     'kwargs': {'moments': True, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

    # Chemical Continues Data (Protein)
    'ChemProtCon': {'columns': ['html$B-Factor of Binding Site', 'html$Cavity Ligandability', 'html$Cavity Volume (A^3)',
                          'html$Cavity % Hydrophobic', 'html$Cavity % Polar', 'rcsb$rcsb_entry_info$molecular_weight',
                           'rcsb$refine_hist$number_atoms_solvent', 'rcsb$refine_hist$number_atoms_total', 
                           'rcsb$refine_hist$pdbx_number_atoms_protein', 'rcsb$rcsb_entry_info$polymer_molecular_weight_maximum', 'rcsb$rcsb_entry_info$polymer_molecular_weight_minimum',
                           'rcsb$rcsb_entry_info$polymer_monomer_count_maximum', 'rcsb$rcsb_entry_info$polymer_monomer_count_minimum',
                           'rcsb$rcsb_entry_info$solvent_entity_count', 'rcsb$rcsb_entry_info$deposited_atom_count', 'rcsb$rcsb_entry_info$deposited_hydrogen_atom_count',
                          'rcsb$rcsb_entry_info$deposited_modeled_polymer_monomer_count', 'rcsb$rcsb_entry_info$deposited_polymer_monomer_count',
                          'rcsb$rcsb_entry_info$deposited_solvent_atom_count', 'rcsb$rcsb_entry_info$inter_mol_covalent_bond_count',
                          'rcsb_prot$entity_poly$rcsb_sample_sequence_length', 'rcsb_prot$rcsb_polymer_entity$formula_weight'],
              'kwargs': {'moments': True, 'primary_count': False, 'count_percent': False, 'secondary_count': False}},

    # Chemical Ordinal Data (Protein)
    'ChemProtOrd': {'columns': ['html$Number of Chains', 'html$Number of Residues in Binding Site',
                          'html$Standard Amino Acids in Binding Site', 'html$Non Standard Amino Acids in Binding Site',
                          'html$Water Molecules in Binding Site', 'rcsb$Number of Chains', 'rcsb$Number of Different Chains',
                          'rcsb$rcsb_entry_info$disulfide_bond_count', 'rcsb$rcsb_entry_info$inter_mol_metalic_bond_count',
                          'rcsb_prot$entity_poly$rcsb_non_std_monomer_count', 'rcsb_prot$rcsb_polymer_entity$pdbx_number_of_molecules'
                          ],
              'kwargs': {'moments': True, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

    # Chemical Categorical Data (Protein)
    'ChemProtCat': {'columns': ['html$Metals in Binding Site', 'html$Cofactors in Binding Site', 'html$SCOPe Chain Classes', 
                          'html$Chain Site Distribution', 'rcsb$Chain Configuration', 'rcsb_prot$entity_poly$rcsb_non_std_monomers',
                          'uniprot$Subcellular Locations', 'uniprot$Organism', "basic$UniRef50"],
              'kwargs': {'moments': False, 'primary_count': True, 'count_percent': True, 'secondary_count': False, 'top_n': 20}},


    # Chemical Continues Data (Ligand)
    'ChemLigCon': {'columns': ['html$Ligand Molecular Weight', 'html$Ligand Buried Surface Area (A^2)', 'html$Ligand Polar Surface Area (A^2)',
                          'rcsb$rcsb_entry_info$nonpolymer_molecular_weight_maximum', 'rcsb$rcsb_entry_info$nonpolymer_molecular_weight_minimum',
                          'rcsb$refine_hist$pdbx_number_atoms_ligand', 'rcsb$refine_hist$pdbx_number_atoms_nucleic_acid',
                          'rcsb_ligand$rcsb_nonpolymer_entity$formula_weight'],
              'kwargs': {'moments': True, 'primary_count': False, 'count_percent': False, 'secondary_count': False}},

    # Chemical Ordinal Data (Ligand)
    'ChemLigOrd': {'columns': ['html$Ligand H-Bond Acceptors', 'html$Ligand H-Bond Donors', 'html$Ligand Rings', 'html$Ligand Aromatic Rings',
                          'html$Ligand Anionic Atoms', 'html$Ligand Cationic Atoms', 'html$Ligand Rule of Five Violation',
                          'html$Ligand Rotatable Bonds', 'rcsb_ligand$rcsb_nonpolymer_entity$pdbx_number_of_molecules'],
              'kwargs': {'moments': True, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

    # Chemical Categorical Data (Ligand)
    'ChemLigCat': {'columns': ['html$Ligand Formula', 'rcsb_ligand$pdbx_entity_nonpoly$comp_id', 'rcsb_ligand$pdbx_entity_nonpoly$name',
                               'html$Ligand Het Code','html$Ligand DrugBank ID'],
              'kwargs': {'moments': False, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

    # Meta Data Continuous Values
    'MetCon': {'columns': ['html$Resolution', 'html$Deposition Date', 'rcsb$cell$length_a', 'rcsb$cell$length_b', 'rcsb$cell$length_c',
                          'rcsb$diffrn$ambient_temp', 'rcsb$exptl_crystal_grow$p_h', 'rcsb$diffrn_source$pdbx_wavelength',
                          'rcsb$rcsb_entry_info$resolution_combined', 'rcsb_prot$entity_poly$rcsb_mutation_count',],
                     'kwargs': {'moments': True, 'primary_count': False, 'count_percent': False, 'secondary_count': False}},

    # Meta Data Ordinal Values
    'MetOrd': {'columns': [],
                     'kwargs': {'moments': True, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

    # Meta Data Categorical Values
    'MetCat': {'columns': ['html$Method', 'rcsb$exptl$method', 'html$Organism', 'html$Reign', 'rcsb$diffrn_detector$detector', 'rcsb$diffrn_radiation$pdbx_diffrn_protocol',
                          'rcsb$diffrn_source$source', 'rcsb$Dominant Chain Found', 'rcsb$Main Ligand Found', 'rcsb$struct_keywords$pdbx_keywords',
                          'rcsb$refine$pdbx_method_to_determine_struct', 'rcsb$exptl_crystal_grow$method',  'rcsb_prot$entity_src_gen$gene_src_genus',
                          'rcsb_prot$entity_src_gen$host_org_genus', 'rcsb_prot$entity_src_gen$pdbx_gene_src_scientific_name', 'rcsb_prot$entity_src_gen$pdbx_host_org_scientific_name',],
                     'kwargs': {'moments': False, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

     # Mol2 Overview Continues
    'Mol2OverCon': {'columns': ['mol2$Protein Number Of Atoms', 'mol2$Protein Number Of Bonds', 'mol2$Protein Number Of Substructures', 'mol2$Protein Number Of Residues',
                             'mol2$Protein Cubic Volume', 'mol2$Protein Molecular Weight', 'mol2$Protein Number Of Binding Residues', 'mol2$Protein Percent Of Binding Residues'],
                     'kwargs': {'moments': True, 'primary_count': False, 'count_percent': False, 'secondary_count': False}},

     # Mol2 Overview Ordinal
    'Mol2OverOrd': {'columns': ['mol2$Protein Number Of Chains', 'mol2$Protein Number Of Different Chains'],
                     'kwargs': {'moments': True, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},

                    
     # Mol2 Overview Categorical
    'Mol2OverCat': {'columns': ['mol2$Protein Chain Configuration'],
                     'kwargs': {'moments': False, 'primary_count': True, 'count_percent': True, 'secondary_count': False, 'top_n': 100}},


     # Mol2 Individual Elements (As Continuous)
    'Mol2ElemCon': {'columns': [c for c in df.columns if 'mol2$' in c and 'Atom' in c and not 'Type' in c and not 'Charge' in c and not 'Number' in c],
                     'kwargs': {'moments': True, 'primary_count': False, 'count_percent': False, 'secondary_count': False}},

     # Mol2 Residues Statistics (As Continuous)
    'Mol2ResCon': {'columns': [c for c in df.columns if 'mol2$' in c and 'Structure' in c and any(res in c for res in SCPDB.RESIDUES)],
                     'kwargs': {'moments': True, 'primary_count': False, 'count_percent': False, 'secondary_count': False}},

     # Mol2 Residues Statistics (As Ordinal)
    'Mol2StrOrd': {'columns': [c for c in df.columns if ('mol2$' in c) and ('Structure' in c) and ('Type' not in c) and ('Count' in c)
                              and ('Ligand' not in c) and ('Site' not in c) and not any(res in c for res in SCPDB.RESIDUES) and 
                              (not any(a in c for a in ['CA', 'NI', 'FE', 'MN', 'ZN', 'NA', 'CL', 'IOD', 'MG', 'HG', 'CO', 'CD'])) and df[c].notna().sum() > 50],
                     'kwargs': {'moments': True, 'primary_count': True, 'count_percent': True, 'secondary_count': False}},
}

def describe_one_column(c: str,
                        df: pd.DataFrame,
                        primary_count: bool = True,
                        top_n: int = 5,
                        lower_limit: int = 1,
                        count_percent: bool = False,
                        cutoff_mode: str = 'top',
                        secondary_count: bool = False,
                        moments: bool = True,
                        print_on: bool = False) -> dict:
    """
    Computes an aggregated descriptor of a column "c" of dataframe "df"

    Args:
        c (str): column name
        df (pd.DataFrame): dataframe
        primary_count (bool, optional): count values. Defaults to True.
        top_n (int, optional): return top n most frequent values. Defaults to 5.
        lower_limit (int, optional): return value counts more than lower limit. Defaults to 1.
        count_percent (bool, optional): also return percetage of values. Defaults to False.
        cutoff_mode (str, optional): how to decide on cutoff of shown values ('top' or 'limit'). Defaults to 'top'.
        secondary_count (bool, optional): count values over first-order count. Defaults to False.
        moments (bool, optional): descriptive statistics. Defaults to True.
        print_on (bool, optional): print the report. Defaults to False.

    Returns:
        dict: descriptor
    """

    assert isinstance(c, str), f'c must be a str, not {type(c)}'
    assert isinstance(df, pd.DataFrame), f'df must be pd.DataFrame, not {type(df)}'
    assert c in df.columns, f'{c} is not within df columns'
    assert isinstance(top_n, int), f'top_n must be int, not {type(top_n)}'
    assert top_n > 0, f'top_n must be positive, not {top_n}'
    assert isinstance(lower_limit, int), f'lower_limit must be int, not {type(lower_limit)}'
    assert isinstance(cutoff_mode, str), f'cutoff_mode must be a str, not {type(cutoff_mode)}'
    assert cutoff_mode in {'top', 'limit'}, f'cutoff_mode can only be "top" or "limit", not {cutoff_mode}'
    assert isinstance(primary_count, bool), f'primary_count must be a bool, not {type(primary_count)}'
    assert isinstance(secondary_count, bool), f'secondary_count must be a bool, not {type(secondary_count)}'
    assert isinstance(moments, bool), f'moments must be a bool, not {type(moments)}'
    assert isinstance(print_on, bool), f'print_on must be a bool, not {type(print_on)}'
    assert isinstance(count_percent, bool), f'count_percent must be a bool, not {type(count_percent)}'

    descriptor = {}
    report = ''

    # Parse column name

    if '$' not in c:
        source = 'combined'
        category = c
    else:
        source = ' '.join(c.split('$')[:-1])
        category = c.split('$')[-1]
    header = f'Source: {source}; Category: {category};'
    descriptor.update({'Source': source, 'Category': category})
    report += header + '\n'

    # Coverage
    non_empty = df[c].replace("None", np.nan).count()
    descriptor.update({'Filled': non_empty})
    descriptor.update({'Filled%DF': f'{100 * non_empty / len(df):.3g} %'})
    report += f'Non empty entries: {non_empty} ({100 * non_empty / len(df):.3g} % of DF total)' + '\n'

    # Unique
    n_unique = df[c].nunique()
    descriptor.update({'Unique': n_unique})
    descriptor.update({'Unique%DF': f'{100 * n_unique / len(df):.3g} %', 'Unique%Filled': f'{100 * n_unique / non_empty:.3g} %'})
    report += f'Unique entries: {n_unique} ({100 * n_unique / len(df):.3g} % of DF total) ({100 * n_unique / non_empty:.3g} % of non empty)' + '\n'

    # Primary count
    if n_unique < non_empty:
        value_counts = df[c].value_counts()
        value_counts_df = pd.DataFrame({'Value': value_counts.index, 'Frequency': value_counts})
        value_counts_df.reset_index(inplace=True)

        # Primary count
        values_to_show = []
        for i, row in value_counts_df.iterrows():

            if cutoff_mode == 'top':
                if i > top_n:
                    break
            elif cutoff_mode == 'limit':
                if int(row['Frequency']) < lower_limit:
                    break
            else:
                raise ValueError('Unknown cutoff method, consider top / limit')
            
            v = f'{row["Value"]} ({int(row["Frequency"])}'
            if count_percent:
                v += f'/{100 * int(row["Frequency"]) / len(df):.3g}%)' # ({100 * int(row["Frequency"]) / non_empty:.3g} %F)
            else:
                v += ')'
            values_to_show.append(v)

        if primary_count:
            report += f'Most common values (cutoff by {cutoff_mode}: {top_n if cutoff_mode == "top" else lower_limit}):' + '\n'
            report += ("; " if not count_percent else "\n").join(values_to_show) + '\n'
            descriptor.update({'PrimaryCount': "; ".join(values_to_show)})

        # Secondary count
        if secondary_count:
            n_frequencies = value_counts_df['Frequency'].nunique()
            report += f'There are {n_frequencies} frequency groups' + '\n'
            secondary_counts_df = value_counts_df.groupby('Frequency', as_index=False).count()
            secondary_counts_df = secondary_counts_df.sort_values(by='Value', ascending=False)
            secondary_counts_df = secondary_counts_df.reset_index()
            secondary_values_to_show = []
            secondary_counts_df.sort_values(by='Frequency', ascending=False, inplace=True)
            for i, row in secondary_counts_df.iterrows():
                
                secondary_values_to_show.append(f'Encountered {row["Frequency"]} times ({row["Value"]})')
            report += 'Most common secondary values:' + '\n'
            report += "; ".join(secondary_values_to_show[:top_n]) + '\n'
            descriptor.update({'SecondaryCount': "; ".join(secondary_values_to_show[:top_n])})

    else:
        report += 'All non-empty values are unique' + '\n'

    # Moments

    if moments:

        nan_free = df.loc[~df[c].isna(), c]
        try:
            minimum = nan_free.min()
            report += f'Minimum value: {minimum}' + '\n'
            descriptor.update({'Min': minimum})
        except TypeError:
            pass
        try:
            maximum = nan_free.max()
            report += f'Maximum value: {maximum}' + '\n'
            descriptor.update({'Max': maximum})
        except TypeError:
            pass
        try:
            mean_ = nan_free.mean()
            report += f'Mean value: {mean_}' + '\n'
            descriptor.update({'Mean': mean_})
        except TypeError:
            pass
        try:
            median_ = nan_free.median()
            report += f'Median value: {median_}' + '\n'
            descriptor.update({'Median': median_})
        except TypeError:
            pass
        try:
            range_ = maximum - minimum
            report += f'Range: {range_}' + '\n'
            descriptor.update({'Range': range_})
        except TypeError:
            pass
        try:
            std = nan_free.std()
            report += f'SD: {std}' + '\n'
            descriptor.update({'Std': std})
        except TypeError:
            pass
    
    report += '\n'

    if print_on:
        print(report)
    
    return descriptor

def describe_database(df: pd.DataFrame,
                      group_specs: dict = None) -> dict[str:pd.DataFrame]:
    """
    Uses group specification to aggregate df columns and make several dfs

    Returns:
        dict: dictionary of DataFrames
    """
    
    assert isinstance(df, pd.DataFrame), f'df must be pd.DataFrame, not {type(df)}'
    
    if group_specs is None:
        group_specs = COLUMN_GROUPS(df)
    else:
        assert isinstance(group_specs, dict), f'group_specs must be dict, not {type(group_specs)}'
    

    tables = {}

    for group, group_dict in group_specs.items():

        kws = group_dict['kwargs']
        columns = group_dict['columns']

        t = []
        for c in columns:
            t.append(describe_one_column(c, df=df, **kws))
        
        t = pd.DataFrame(t)

        tables.update({group: t})

    elements = ['H', 'C', 'N', 'O', 'S', 'P', 'Se', 'Mg', 'Zn', 'Fe', 'Ca', 'Mn', 'Cu', 'Co', 'Ni', 'Na', 'K', 'F', 'Cl', 'Br', 'I', 'Du']

    t: pd.DataFrame = tables['Mol2ElemCon'].copy()
    t['Element'] = t['Category'].map(lambda x: elements.index(x.split(' ')[-1]))
    t['Metric'] = t['Category'].map(lambda x: 'Count' in x)
    t['Scope'] = t['Category'].map(lambda x: not 'Chain' in x)
    t['Object'] = t['Category'].map(lambda x: ('Site' in x) * 2 + ('Protein' in x) * 1 + ('Ligand' in x) * 3)

    t = t.loc[(t['Object'] != 3) & t['Scope']]
    t = t.sort_values(by=['Scope', 'Element', 'Metric', 'Object'])

    new_table = []
    for i in range(len(elements)):

        if elements[i] in ['Se', 'Co', 'Ni', 'Du', 'F']:
            continue
        new_row = {
            'Element': elements[i],
            'N prot': t.loc[(t['Element'] == i) & t['Metric'] & (t['Object'] == 1), 'Mean'].to_list()[0],
            'N site': t.loc[(t['Element'] == i) & t['Metric'] & (t['Object'] == 2), 'Mean'].to_list()[0],
            '% prot': t.loc[(t['Element'] == i) & (~t['Metric']) & (t['Object'] == 1), 'Mean'].to_list()[0],
            '% site': t.loc[(t['Element'] == i) & (~t['Metric']) & (t['Object'] == 2), 'Mean'].to_list()[0],
        }
        new_table.append(new_row)

    t_ = pd.DataFrame(new_table)
    t_['Source'] = 'MOL2'
    tables.update({'Mol2ElemShort': t_})

    t: pd.DataFrame = tables['Mol2ResCon'].copy()
    t['R'] = t['Category'].map(lambda x: x.split(' ')[-1])
    t['Metric'] = t['Category'].map(lambda x: 'Count' in x)
    t['Scope'] = t['Category'].map(lambda x: not 'Chain' in x)
    t['Object'] = t['Category'].map(lambda x: ('Site' in x) * 2 + ('Protein' in x) * 1 + ('Ligand' in x) * 3)

    t = t.loc[(t['Object'] != 3) & t['Scope']]
    t = t.sort_values(by=['Scope', 'R', 'Metric', 'Object'])

    new_table = []
    for r in SCPDB.RESIDUES:

        new_row = {
            'R': r,
            'N prot': t.loc[(t['R'] == r) & t['Metric'] & (t['Object'] == 1), 'Mean'].to_list()[0],
            'N site': t.loc[(t['R'] == r) & t['Metric'] & (t['Object'] == 2), 'Mean'].to_list()[0],
            '% prot': t.loc[(t['R'] == r) & (~t['Metric']) & (t['Object'] == 1), 'Mean'].to_list()[0],
            '% site': t.loc[(t['R'] == r) & (~t['Metric']) & (t['Object'] == 2), 'Mean'].to_list()[0],
        }
        new_table.append(new_row)

    tt_ = pd.DataFrame(new_table)
    tt_['Source'] = 'MOL2'
    tables.update({'Mol2ResShort': tt_})
    return tables

class Dichotomizer():

    """
    Main ideas:
    1. Take a rectangle and a list of data
    2. Divide the data so that there are 2 groups of approximately equal sum.
    3. Divide the long side of a rectangle proportional to the sums of two sub-groups
    4. Call the class again with new starting rectangle and sub-group as new list.
    5. Repeat recursively until there is a single element in the data - return coordinates of the final rectangle.

    """

    def __init__(self, lower_left_corner: tuple[float, float], upper_right_corner: tuple[float, float], depth: int = 0):
        self.x1 = lower_left_corner[0]
        self.y1 = lower_left_corner[1]
        self.x2 = upper_right_corner[0]
        self.y2 = upper_right_corner[1]

        self.data = None
        self.terminal = None
        self.depth = depth

    def fit_data(self, data: list, register_of_lines: list = []):

        self.data = data
        sum = np.sum(self.data)


        if len(self.data) == 1:
            self.terminal = True
            return [(self.x1, self.y1, self.x2, self.y2, self.data, self.depth)]
        
        sum1, factor = 0, 0
        for i, x in enumerate(self.data):
            sum1 += x
            if sum1 / sum >= 0.5:
                factor = i + 1
                break

        if factor >= len(self.data):
            factor = len(self.data) - 1

        data1, data2 = self.data[:factor], self.data[factor:]
        sum, sum1, sum2 = np.sum(self.data), np.sum(data1), np.sum(data2)
        delta = 0

        if self.x2 - self.x1 > self.y2 - self.y1:
            split_point = self.x1 + (sum1 / sum) * (self.x2 - self.x1)
            split_line = ((split_point, self.y1), (split_point + delta, self.y2))
            sub_field1 = Dichotomizer(lower_left_corner=(self.x1, self.y1), upper_right_corner=(split_point, self.y2), depth = self.depth + 1)
            sub_field2 = Dichotomizer(lower_left_corner=(split_point + delta, self.y1), upper_right_corner=(self.x2 + delta, self.y2), depth = self.depth + 1)
        else:
            split_point = self.y1 + (sum1 / sum) * (self.y2 - self.y1)
            split_line = ((self.x1, split_point), (self.x2, split_point + delta))
            sub_field1 = Dichotomizer(lower_left_corner=(self.x1, self.y1), upper_right_corner=(self.x2, split_point), depth = self.depth + 1)
            sub_field2 = Dichotomizer(lower_left_corner=(self.x1, split_point + delta), upper_right_corner=(self.x2, self.y2 + delta), depth = self.depth + 1)
            
        register_of_lines.append((split_line, self.depth))
        sub_areas1 = sub_field1.fit_data(data1, register_of_lines)
        sub_areas2 = sub_field2.fit_data(data2, register_of_lines)
        return [*sub_areas1, *sub_areas2]


def adjust_text(s: str, x1: float, x2: float) -> str:
    """
    A small utility to make annotations fit better into rectangles. Not perfect but better than default behaviour of plt text.

    Args:
        s (str): string to refactor
        x1 (float): smaller coordinate of a rectangle
        x2 (float): larger coordinate of a rectangle

    Returns:
        str: refactored text
    """
    s_ = s[:]
    s_ = s_.replace('(', ' (')
    s_ = re.split('([- ])', s_)
    max_length = np.max([len(x) for x in s_]) + 2
    width = x2 - x1
    fontsize = min(50 * width / max_length, 30)
    result = ''
    for i, part in enumerate(s_):
        if part == '-':
            result += part
        elif part == ' ':
            continue
        else:
            if len(result.split('\n')[-1]) + len(part) + 1 <= max_length:
                result += ' ' * (i != 0) + part
            else:
                result += '\n' * (i != 0) + part
    return result, fontsize

def construct_enclosure(df: pd.DataFrame,
                        column_of_values: str,
                        column_of_annotations: str,
                        column_of_attribute: str,
                        title: str = None,
                        dim1: int = 20,
                        dim2: int = 20,
                        colormap: str | dict = None,
                        figsize: int = 16,
                        alpha: float = 1.0,
                        save: str = None) -> list[tuple]:
    """
    Main function which calles recursive dichotomizer and also
    organize attribute values into bines and plot textual annotations.

    Args:
        df (pd.DataFrame): pandas DataFrame with all the necessary data
        column_of_values (str): name of the column which will be used to create rectangles
        column_of_annotations (str): name ofthe  column which will be used to add text on the plot
        column_of_attribute (str): name of the column, whcih will be used for coloring
        title (str, optional): title of the plot - can be generated automatically. Defaults to None.
        dim1 (int, optional): horizontal dimension. Defaults to 20.
        dim2 (int, optional): vertical dimension. Defaults to 20.
        colorcmap (str | dict, optional): explicit dictionary. Defaults to None.

    Returns:
        list[tuple]: returns a list of areas and values
    """

    necessary_columns = [column_of_values, column_of_attribute, column_of_annotations]
    for c in necessary_columns:
        assert isinstance(c, str), f'column name must be a string, not {type(c)} of {c}'
        assert c in df.columns, f'no column named "{c}" in the provided dataframe'

    assert isinstance(title, str), f'title must be a string, not {type(title)}'
    assert isinstance(dim1, int) and isinstance(dim2, int) and dim1 > 0 and dim2 > 0, f'dim1 and dim2 must be positive integers, not ({type(dim1)}, {type(dim2)})'
    assert isinstance(colormap, dict), f'colorcmap must be a dict, not {type(colormap)}'
    

    data_ = df[necessary_columns]
    data_ = data_.sort_values(by=[column_of_attribute, column_of_annotations])
    data = data_[column_of_values].to_list()
    annotations = data_[column_of_annotations].to_list()
    attribute = data_[column_of_attribute].to_list()
    k = figsize / max(dim1, dim2)

    def get_color(value):
        return colormap.get(value, None)
    
    grouped_df = df.groupby(column_of_attribute, as_index=False).sum()
    dichotomizer = Dichotomizer((0, 0), (dim1, dim2))
    super_areas = dichotomizer.fit_data(grouped_df[column_of_values].to_list())

    areas = []
    for super_area, group_name in zip(super_areas, grouped_df[column_of_attribute]):
        dichotomizer = Dichotomizer((super_area[0], super_area[1]), (super_area[2], super_area[3]))
        sub_areas = dichotomizer.fit_data(df.loc[df[column_of_attribute] == group_name, column_of_values].tolist())
        areas.extend(sub_areas)

    # dichotomizer = Dichotomizer((0, 0), (dim1, dim2))
    # areas = dichotomizer.fit_data(data)

    fig, axs = plt.subplots(1, 2, figsize=(dim1 * k * 1.2, dim2 * k), width_ratios=[1.0, 0.2])
    ax  = axs[0]
    ax.set_title(title, fontdict={'fontsize': 15})
    for i, a in enumerate(areas):
        s = (a[2] - a[0]) * (a[3] - a[1])
        point1 = (a[0], a[1])
        point2 = (a[2], a[1])
        point3 = (a[2], a[3])
        point4 = (a[0], a[3])

        ax.add_patch(plt.Polygon([point1, point2, point3, point4], color=get_color(attribute[i]), ec='white', alpha=alpha))

        if s / (dim1 * dim2) > 0.005:
            text, font = adjust_text(annotations[i], a[0], a[2])
            ax.text(x = 0.5 * (a[0] + a[2]), y = 0.5 * (a[1] + a[3]), s = text, c='white', fontsize=font,
                    horizontalalignment='center',
                    verticalalignment='center')
    
    ax.set_ylim(0, dim2 + 0.5)
    ax.set_xlim(0, dim1)
    ax.axis('off')


    ax_legend = axs[1]
    ax_legend.axis('off')
    # ax_legend.set_title('Legend:')

    handles = [
        Patch(facecolor=color, edgecolor='white', label=str(label), alpha=alpha)
        for label, color in colormap.items()
    ]

    ax_legend.legend(
        handles=handles,
        loc='center',
        ncol=1,
        frameon=False
    )

    if save is not None:
        fig.savefig(save)
    
    plt.show()

    return areas
