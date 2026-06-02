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

## Abstract

Accurate prediction of protein binding sites is an important step in structure-based drug discovery. It allows for identification of protein regions that are accessible for ligands (drugs), which in turn facilitates the target molecule characterization and virtual screen- ing and eventually makes drug discovery more rational and time-efficient. Although deep learning methods are applied successfully for the task and achieve strong performance, the studies usually evaluate models using heterogeneous datasets without analyzing how the performance varies across different types of proteins.

This study leverages the generic but robust and well-established Equivariant Graph Neu- ral Network (EGNN) model with a preprocessing pipeline inspired by DeepSite and inves- tigates the dependence of its performance with respect to different protein classifications. As a benchmark dataset, scPDB is chosen, the analysis is conducted on residue-level, and evaluation is performed using cross-validation. Research focuses on structural, functional, and evolutionary groupings and assesses how these properties influence the model’s gen- eralization ability.

Substantial differences in the quality of the predictions are revealed, especially across structural classes. Protein size exhibits the strongest effect on the performance: medium- sized proteins (100-400 residues) consistently outperform both smaller and larger pro- teins. Considering the overall structural composition, proteins with spatially separated α-helices and β-sheets outperform structures with interleaved motifs (α/β) and multi- domain proteins. Membrane proteins and small proteins with an irregular secondary structure perform significantly worse than other groups. In contrast, the observed effects of functional and evolutionary classifications are weaker and less consistent than those of structural groupings.

These findings demonstrate that structural heterogeneity affects the performance and generalization of EGNN-based models of binding site prediction. Several directions for future research can be proposed according to the observations: using separate models for proteins of different sizes and structural compositions and applying individual prepro- cessing strategies for structurally atypical samples.
