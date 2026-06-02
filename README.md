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
