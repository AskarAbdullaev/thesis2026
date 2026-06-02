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

├── all_scpdb_entries.txt
├── excluded_scpdb_entries.txt
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
│   ├── SCOPe
│   │   └── 2_08.csv
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
│   ├── scPDB
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

Missing directories:
	None

Missing files:
	None
./
|- final_linear.pt (final linear model)
|- final_cnn.pt (final CNN model)
|
|- all_scpdb_entries.txt (list of all scPDB entries as of Sep.'25)
|- excluded_scpdb_entries.txt (entries that are filtered out during preprocessing)
|- decode.npy (decoder of atom properties)
|- original_split.txt (data split provided in DeepSite supplementary files)
|
|- data_utilities.py (utilities for the 'load_and_process' notebook)
|- load_and_process.ipynb (notebook for data loading and preprocessing)
|
|- sample_and_train_utilities.py (utilities for 'sample_and_train' and 'cross_validation' notebooks)
|- sample_and_train.ipynb (notebook for voxelization, sampling and hyperparameters search)
|
|- cross_validation.ipynb (notebook for cross-validation and final training)
|
|- inference_utilities.py (utilities for the 'inference.ipynb' notebook)
|- inference.ipynb (notebook with the final inference and domain-specific metrics)
|
|- Data 
    |
    |- scPDB (original scPDB dataset)
        |- 1a2b_1
            |- protein.mol2
            |- site.mol2
            |- ...(other possible files from scPDB)
        |- ...(all scPDB entries)
    |
    |- SCOPe (folder with SCOPe versions)
        |- 2_08.csv
        |- ...(possibly other SCOPe versions)
    |
    |- Pages (textual files with scPDB web-pages source codes)
        |- 1a2b_1.txt
        |- ...(all the source codes of scPDB web pages)
    |
    |- Folds (CSVs with data split / folds)
        |- 1.csv
        |- ...(other folds)
        |- test.csv
    |
    |- Atoms (CSVs with processed atoms, including chemical channels)
        |- 1a2b_1.csv
        |- ...(csvs with atom coords and props)
    |
    |- Voxels (voxelized entries in compact form: atom_grid + occupancy + decoder (common))
        |- 1
            |- 1a2b_1
                |- atoms_grid.npy
                |- occupancy.npy
                |- site_center.npy
            |- ... (other voxelized entries with voxel size 1)
        |
        |- 2
            |- 1a2b_1
                |- atoms_grid.npy
                |- occupancy.npy
                |- site_center.npy
            |- ... (other voxelized entries with voxel size 2)
    |
    |- CV (Training logs)
        |- analysis (folder with aggregated CSVs)
        |- pilot (folder for hyperparameter search)
            |- bs_128_do_0_cs_1
                |- train_loss.txt
                |- test_loss.txt
                |- true_labels.npy
                |- predictions.npy
            |- ... (other hyperparameter combinations)
        |
        |- vs_1_cnn
            |- params.csv (parameters of the run)   
            |- 0
                |- train_loss.txt
                |- test_loss.txt
                |- true_labels.npy
                |- predictions.npy
            |- ... (other folds)
        |
        |- ... (other models)
    |
    |- Inference (folder with subgrid scores)
        |- cnn
            |- 1a4z_4.npy
            |- (...other entries from the test set)
        |
        |- linear
            |- 1a4z_4.npy
            |- (...other entries from the test set)
    |
    |- final_metrics.csv
```

## Introduction
