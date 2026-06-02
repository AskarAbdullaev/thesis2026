# thesis2026
Thesis Research 2026


## Dependencies

For this project I used the following modules:

| Module         | Version |
| -------------- | ------- |
| Python version | 3.12.13 |
| requests       | 2.33.1  |
| pandas         | 3.0.1   |
| numpy          | 2.4.3  |
| regex          | 2026.2.28 |
| tqdm           | 4.67.3  |
| bs4            | 4.14.3  |
| matplotlib     | 3.10.8   |
| torch          | 2.11.0+cu128   |
| sklearn        | 1.7.1   |
| scipy          | 1.17.1  |
| seaborn        | 0.13.2 |
| torch_geometric | 2.7.0 |

## File structure

```text

├── all_scpdb_entries.txt  (list of all scPDB entries as of Sep.'25)
├── excluded_scpdb_entries.txt (entries that are filtered out during preprocessing)
├── 01_loading.ipynb
├── 02_data.ipynb
├── 03_preprocessing_and_hp.ipynb
├── 04_train_and_analyze.ipynb
├── utilities.py
├── initial_tools.py
├── data_tools.py
├── scpdb.py
├── preprocessing.py
├── dataset.py
├── gnn.py
├── training.py
├── cv_logs.csv
├── Files
│   ├── SCOPe (folder with SCOPe versions)
|   |   ├── (possibly other SCOPe versions)
│   │   └── 2_08.csv
│   ├── Residues_Only_25_08 (graph instances precomputed)
|   |   ├── 1a2b_1
|   |   |    ├── protein_features.npy
|   |   |    ├── resonly_1a2b_1_res_ALA15_A_1.npy
|   |   |    ├── resonly_1a2b_1_res_ALA15_A_1_edges.npy
│   │   |    └── ... (all other subgraphs)
│   │   └── ...(all other entries)
│   ├── Folds
│   │   ├── 0.txt
│   │   ├── 1.txt
│   │   ├── 2.txt
│   │   ├── 3.txt
│   │   ├── 4.txt
│   │   ├── 5.txt
│   │   ├── 6.txt
│   │   ├── 7.txt
│   │   ├── 8.txt
│   │   ├── 9.txt
│   │   ├── 101.txt
│   │   ├── 102.txt
│   │   └── 103.txt
│   ├── scPDB (original scPDB dataset + metadata)
│   │   ├──  1a2b_1
|   |   |    ├── protein.mol2
|   |   |    ├── ligand.mol2
|   |   |    ├── site.mol2
|   |   |    ├── cavity.mol2
|   |   |    ├── html.txt
|   |   |    ├── html.json
|   |   |    ├── P61586.json
|   |   |    ├── assembly_1.json
|   |   |    ├── identifiers.json
|   |   |    ├── rcsb_entry.json
|   |   |    ├── rcsb_entity_1.json
|   |   |    ├── rcsb_entity_2.json
|   |   |    ├── FASTA.txt
|   |   |    ├── FASTAsite.txt
|   |   |    ├── atoms.csv
|   |   |    ├── residues.csv
│   │   |    └── ... (other possible files)
│   │   └── ...(all other entries)
│   └── database_v0.csv
├── SIFTS
│   ├── pdb_chain_enzyme.csv
│   ├── pdb_chain_uniprot.csv
│   ├── scop.csv
│   └── scop_names.csv
├── CV
│   ├── 0.csv
│   ├── 1.csv
│   ├── 2.csv
│   ├── 3.csv
│   └── 4.csv
└── Metrics
    ├── ec_bulk.csv
    ├── ec_per_entry.csv
    ├── organism_bulk.csv
    ├── organism_per_entry.csv
    ├── reign_bulk.csv
    ├── reign_per_entry.csv
    ├── scope_bulk.csv
    ├── scope_per_entry.csv
    └── per_entry.csv

 
```

## Introduction
