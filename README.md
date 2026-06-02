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

## Introduction
Proteins and Ligands. Proteins are large complex molecules composed of sequences of 20 different amino acids, which are arranged in one or more polypeptide chains. Proteins are responsible for a wide variety of functions in a living organism, such as catalysis, structural integrity, transport, signaling, immune response, etc. [53]. Drugs are typically small molecules that are capable of producing biological effects when delivered to the organism. One of the common mechanisms includes a drug binding to a protein in order to modify its activity [56]. An interaction between a drug (ligand) and a protein (target) is usually non-covalent and occurs in a specific region of a protein called a ’binding site’ or a ’binding pocket’ (see Figure 1, Left). Such an interaction can inhibit enzymatic activity, destabilize conformation, alter signaling processes, etc. Thus, identifying binding sites helps to understand the way a protein functions, the properties of potential ligands, and is an essential step for efficient drug design [56].

<img width="4000" height="3000" alt="1dmp" src="https://github.com/user-attachments/assets/4b8c914d-8596-4dcc-a202-31692f0e5be6" />
Figure 1. HIV-1 protease (blue) complex with DMQ/Mozenavir (yel-
low) resolved in the pocket (red). PDB: 1dmp.


Binding Sites. The three-dimensional structures of proteins form intricate surfaces with numerous concavities and protrusions, creating distinct microenvironments for ligand binding and catalysis [48]. Therefore, binding sites are typically localized regions com- posed of several residues. These regions are particularly favorable for recognition and interaction with specific ligands. There might also be more than one binding site in a protein. In some cases, a binding site is responsible for the regulation of protein function - binding the ligand leads to conformational changes in the entire protein, modifying its activity. Such sites are called allosteric [53]. However, since experimental datasets, such as scPDB, often stem from crystallography, a common approach is to consider binding pockets as local environments.

Drug Discovery. There exist several paradigms in drug discovery[83]: the phenotypic screening approach analyzes observable effects without researching molecular structures and mechanisms; ligand-based methods rely on the already known activity information, structures and characteristics of existing drugs in order to invent novel ones. In contrast, structure-based drug discovery (SBDD) leverages knowledge of target molecules - usually proteins - to select or design possible ligands that would interact with a target [32]. The initial steps of a classical SBDD include identifying the target protein, determining po- tentially druggable binding pockets, performing virtual screening and docking algorithms (in silico), and detecting candidates [83] (see Figure 1, Right).

Deep Learning. The experimental research of protein-ligand interactions is an expen- sive and time-consuming task [81]. Thus, in recent years, computational approaches have been increasingly adopted by the industry to improve the efficiency and effectiveness of the drug discovery process [81]. Noticeably, deep learning has become a powerful tool in computational biology [3]. In 2017, DeepSite[30] suggested applying Convolutional Neural Network for the protein segmentation task. The method reached state-of-the-art performance in binding site prediction, and more CNN-based tools appeared in sub- sequent years [40, 13, 63, 52]. Later, GraphBind [74] (2021) successfully implemented Graph Neural Network for a similar task. Since then, a number of prominent GNN-based models have been released [66, 61, 73, 80, 60].

Goal. The main goal of this research is to assess the comparative performance of binding- site prediction across different groups of proteins and to suggest future work based on the findings. To reach this goal, the performance comparison requires a suitable frame- work as a basis. The crucial decision for the research is the choice of the underlying model. Although reaching the highest level of performance is not the primary goal, the model architecture must be modern and capable enough to ensure that the findings are reliable and not outdated to derive meaningful suggestions. Meanwhile, the overcom- plicated pipeline or too specialized architecture can make the final results less general, and the interpretability might suffer. Thus, the underlying model should be modern but well-established and both biologically and computationally sound.

In section 1.2, a domain-specific overview of milestone studies and existing methods for binding site prediction is covered. Based on the overview, the relatively recent but well-studied approach is Graph Neural Network (GNN), which is concise and naturally suitable for protein structures. The principles of GNN and the diversity of GNNs are briefly described in section 1.3. For this research, an Equivariant Graph Neural Network (EGNN) model is chosen and implemented according to conventional design strategies.

Despite being built from a limited set of amino acids, the structural and functional diversity of proteins is vast. The methods covered in the domain-specific overview mostly evaluate the binding site prediction performance across datasets as a whole. While some papers, such as DeepSite [30], report grouped performance, it is only mentioned briefly as a secondary consideration.

Implementation. The central idea of this research is to concentrate on how the model performs across different protein groups rather than on achieving the highest perfor- mance metrics. There exist many protein classifications, some of which are summarized in section 1.4. For the current study, four classifications are chosen as main objectives for the analysis: structural classification (SCOPe [19] classes), classification by protein size, functional classification (EC classes), and evolutionary classification (kingdoms of the source species). There are also two supplementary classifications: by source species and by the type of ligand resolved in the binding site. Supplementary classifications contain numerous small groups, which makes it challenging to analyze them statistically.

The current version of the scPDB database [15] is used as benchmark data. The data undergo a stratified split, and after a hyperparameter selection, 5-fold cross-validation is performed, and residue-level predictions are collected. The subsequent analysis includes residue-level statistics to get a general idea of the patterns and F1 scores computed on the protein level. The differences between groups are checked using the Kruskal-Wallis test, followed by Dunn’s post-hoc test. Whenever groups are found to be significantly different, an effect size is estimated using Cliff’s delta. Finally, for every main classification, a latent order is calculated. The code, notebooks and intermediate files for this research can be found via https://github.com/AskarAbdullaev/thesis2026.

Contribution. This work is meant to contribute to a more detailed understanding of the influence of protein properties and metadata on the binding site prediction performance and to propose possible directions for the future refinement of the predictive pipelines in the domain.

## Materials

scPDB Dataset. The primary source of protein structures and related information for the study is scPDB [15]. scPDB is an annotated database of druggable binding sites derived from the Protein Data Bank (PDB) [4]. The scPDB mainly includes informa- tion on small synthetic and natural ligands, along with corresponding high-quality, non- redundant protein binding sites. For more structural information and for later analyses, I also made use of the latest stable release of a SCOPe database [11]: a database that classifies protein chains according to their structure, function, and taxonomy.

By March 15th, 2026, there are 17,594 entries available for download from the official page of scPDB. However, according to the information on the official site, the last stable release (2017) contains 16,034 entries, 4,782 proteins, and 6,326 ligands. In order to collect more robust metadata and get potential insights into data patterns, I have downloaded the scPDB web pages of as many entries as are available. Using the BeautifulSoup4 module [55], I have successfully found 16,032 (91.1% of the database) entries, which is only two less than a stable release contains. Apart from scPDB, I have gathered additional descriptors through RCSB API[57], which provides various information about PDB entries, including citations, refinements, experimental setup, distinct molecules, aggregate metrics, etc. The main API endpoint is available for 17,539 entries (99.7%).

Aggregation is performed over all 17,594 entries, and selected results are displayed in the Appendix via several tables. It is necessary to note that each row in these tables has a ’Source’ column, which indicates the origin of the data (’HTML’ stands for scPDB web page; ’RCSB’ stands for RCSB API; ’MOL2’ stands for structural files from the scPDB archive; and ’BASIC’ stands for the accessibility of entries). As far as each property covers a certain portion of entries while being unknown for the rest, a ’Coverage’ field is provided. From this observation also arises the fact that similar properties originating from different sources might have slightly different aggregated values.

The metadata reveals several trends. The oldest registered structure in the set dates back to 1975, and the newest deposited structure is stamped with 2017, which makes the dataset slightly outdated. More than 99% of the structures have been resolved using X-ray diffraction, while less than 1% have been resolved using solution NMR. As for the X-ray methodology, the main emitters are synchrotrons (74.2%) and rotating anodes (21.1%), producing waves with lengths typically falling into the 1-2Årange. Two prevalent methods of crystal growth for X-ray analysis include the vapor diffusion of hanging drop (43.8%) and the vapor diffusion of sitting drop (20.2%). The mean pH of crystal growth is 6.77 (Std. 1.18). The diffraction detectors are Charge Coupled Devices (CCDs) in 63.9% of entries and an Image Plates in 21%. Thus, the linear resolution is reported to be around 2.1Å(SD 0.4Å), which is useful to know for the preprocessing stage. According to scPDB web pages, the genes of resolved proteins often belong to Homo sapiens (30.7%), Escherichia coli (5.55%), Rattus norvegicus (3.23%), HIV 1 (2.43%), Mycobacterium tuberculosis (2.23%), and Mus musculus (360/2.05%) (overall: 52.2% from eukaryotes, 30.2% from bacteria, 5.4% from viruses, and 3.27% from archaea).

The PDB structures in the dataset are versatile in their composition. More than half of them have 3 or 4 distinct chemical entities, of which 1 is usually allocated for a solvent. However, some entries have up to 23 entities. The majority of entries (86.2%) have a single polymer entity representing a protein, 9.82% feature 2 polymer entities, while the rest have even more. Nucleic acids are rare ligands in the dataset: only about 1.5% of cases, of which DNA is more frequent than RNA. There is also a single case of a ligand being a DNA-RNA hybrid. Although scPDB only allows a single binding site per entry, the original PDB entries typically have more than 1 ligand: 31.7% have 2 ligands, 22.6% have 3 ligands, and only 25.4% feature one ligand initially. There are also rare cases of branched polymers (usually oligosaccharides): less than 2% of the entries have at least one such entity. Despite a certain skew towards simpler composition, there are outliers with up to 14 different peptide chains or up to 15 types of potential ligands in a single assembly.

Protein-related Observations. Protein-related properties reveal multiple important observations. There are only 4688 unique UniProt identifiers, which is 3.7 times less than the number of entries. Even PDB identifiers are not entirely unique in the dataset; 903 PDB codes are encountered twice, 38 codes are encountered 3 times, and ’1w19’ has 4 instances in scPDB (T1P, T2P, T4P, and T5P - similar ligands of lumazine synthase of Mycobacterium tuberculosis). The dataset is therefore structurally redundant; however, similar proteins appear multiple times in pairs with different ligands.
Speaking about overall chemical properties, the average number of non-hydrogen atoms in a protein is around 5800, while the molecular weight ranges from 5 kDa to 452 kDa with a mean of 33.6 kDa (SD 18) (according to the analysis of mol2 files). 86.3% of proteins do not have disulfide bridges in the resolved structure, and 37.5% include metal ions. Also, cis-peptide bonds are rather rare: only 1.79 instances per protein on average, although some entries have more than 90.

The lengths of peptide chains are highly variable. The values span from only 35 residues to several thousand residues (according to mol2 file analysis). However, the median value is 320, and the mean is 345 (SD 160), still implying a large variance. The total number of residues in all chains is 530 in the median and 777 on average. In 10.8% of cases, there is at least a single non-standard monomer in a protein. Among non-standard residues, the most common are MSE (selenomethionine) found in 3.66% of entries; SEP (phosphoserine) found in 1.25% of entries; and PTR (O-phosphotyrosine), found in 0.74% of entries.

The chain composition tends to be simple in most cases. 87.6% of entries can be described with a single FASTA sequence (although the number of identical monomers varies). 9.18% have 2 unlike chain types, 1.89% have 3 unlike chain types, and 1.33% have more than 3 (up to 14). The chain composition is visualized in Figure 6. According to the RCSB API, around 37%, 33%, and 10% are monomers, homodimers, and homotetramers, respectively. The next frequent structure is heterodimer (≈ 5%), homotrimer (≈ 3%), and symmetric heterotetramer (≈ 2%). The other compositions are more complex but rare.
Considering the function, for 56.3% of proteins in the dataset, enzyme codes are known. The most frequent classes are transferases (19.9%), oxyreductases (13%), and hydrolases (9.67%). Among the subclasses, the most numerous are phosphotransferases/kinases (EC 2.7), peptidases (EC 3.4), and CH-OH oxyreductases (EC 1.1). The SCOPe classification is only available for 68% of the entries. The most frequent classes appear to be ’c’ (α + β mostly interleaved), ’d’ (α + β mostly segregated), and ’b’ (mostly β-sheets). The most common subfamily is ’d.144’ (kinase-like motif). Visualized SCOPe distribution is shown in Figure 5.

Binding Site Observations. The further analysis of the data gives a better view of the construction and properties of the binding sites. The average cavity volume in 3 is estimated to be 793, while the average volume of a cubic grid around a protein is 313,000 3. Information from scPDB web pages also suggests that the presence of a hydrophobic and polar surface in the binding cavity is approximately equal.
In the resolved structures, there are typically 1-2 water molecules present in the binding site. Almost 19% of the binding sites contain metal atoms, mostly magnesium (8.88%) and zinc (4.46%). Also, 43% of the binding sites include non-standard residues. Some binding sites include cofactors, though only 7.9%. The most frequent co-factors are NADP+, FAD, NADPH, and NAD+. The percentage of elements in a whole protein and in the binding site also differs: sulfur, phosphorus, and several metals have a much higher con- centration within the binding site.
The mean number of binding residues per protein is 33.6 (SD 11.9). The more robust metric is the percent of binding residues, which is 9.28 (SD 5.41). However, certain outliers have up to 82% of binding residues. The residue ratio is also shifted in the binding site in comparison to that of the whole protein. Phenylalanine, tryptophan, tyrosine, methionine, cysteine, and histidine have a highly elevated chance of being found in the binding site (see Figure 7). It is important to note that 33.8% of the binding sites are constructed using more than one peptide chain. The distribution between chains varies greatly among these 33.8% of cases.
Ligand-related Observations. In the scope of this research, I am not particularly interested in ligand properties. However, they might give hints for the main topic. The average molecular weight of a ligand is 476 Da with a standard deviation of 181. The tiniest ligand, imidazole (C3H4N2, MW=68), appears in 65 entries. Moenomycin is the largest ligand in the dataset, with a molecular weight of around 1.58 kDa, and only appears once. Ligands in the dataset have on average three times as many hydrogen bond acceptors as hydrogen bond donors (8 vs. 3). Half of the ligands have 2 to 3 rings, while structures without rings constitute only around 10% of cases. Anionic groups at physiological pH are 4-5 times more abundant than cationic groups, although both are not present at all in approximately 34% of the ligands. The frequent ligands are FAD (4.96%), NAD (3.54%), ADP (3.38%), and ATP (2.28%). Such significant shares of certain ligand types might mean that a lot of binding sites in the dataset are structurally similar.

Only distinct protein structures are kept. To prevent erroneous evaluation, the dataset has to be reduced: first, I eliminate duplicates in PDB IDs, which leaves 16612 entries. Second, all entries attributed to SCOPe class ’l’ (structural artifacts) are ex- cluded, as well as entries lacking SCOPe matches, as too risky. From the remaining 12689 entries, I remove those that were not found via RCSB API or lack UniProt identi- fiers. Finally, there is a lot of redundancy in UniProt IDs. However, they might be paired with different ligands, which makes them valuable. Thus, I drop duplicates on UniProt- ligand pairs, which results in ... entries. (Note: UniProt IDs of chains within the protein might be different. As a reference, I take the ID of the chain that contains the majority of binding residues).

In total, 9452 entries are therefore left. The samples are split into 6 portions with UniProt IDs distributed without intersections to prevent data leakage. A 5-fold cross-validation is performed while 1 portion is retained for hyperparameter search. Cross-validation is known to be particularly suitable when a disjoint validation set is not feasible.

## Methodology

Locality Assumption. An important assumption for this research is that binding sites can be approximated as local environments. The scale of meaningful interactions stems from the amino acid geometries. The length of amino acids can vary from 3.9Åfor glycine up to 11.3Åfor lysine, while widths vary between 3.5Åand 6.1Å[53]. Practically, the mean- ingful interaction distance is commonly referred to as 5-6Å.

Design Decisions. Given that the main goal of this research is to analyze the differences in performance across protein groups rather than to focus on reaching state-of-the-art performance for the binding site prediction, I intentionally avoid complex or sophisticated designs. Another crucial consideration that guides the methodological and architectural choices is the amount of computational resources available. The following design choices are therefore imposed:

1. Features passed to the model are extracted on the residue level rather than on the atomic level. Although atomic information is richer, it increases the computational cost and potentially the variance.
2. 
3. Initially, the model had to learn residue embeddings instead of using precomputed features. However, several preliminary experiments (which are left beyond the dis- cussion of this research) revealed that the convergence is too slow under the given limitations. Thus, the approach is changed towards precomputed features in the ac- tual model.
Residue Features. The input data consists of ’mol2’ files, which describe proteins and binding sites on the atomic level. Residues are extracted from the substructures section of TRIPOS notation. Only the 20 standard proteinogenic types are considered. The reason for dropping rare residues is the lack of reliable descriptors. Each residue is represented by a set of predefined features, including length (Å), width (Å), volume (Å3), pKa constant, charge, hydropathy, etc. The exact values are listed in Table 3. Additionally, a one-hot encoding vector is concatenated with the features. In total, each residue is described by 33 scalar values. For the sake of training stability, the fixed features of 20 residue types are normalized column-wise.

Concerning protein aggregate features, they are computed once for the whole protein and include 38 basic descriptors such as percentage of residues of each type, percentage of residues grouped by common chemical features, density of residues, density of atoms, the overall protein shape as a ratio between coordinate span to the diagonal of the coordinate box, etc. Features like molecular weight or total number of residues are very versatile. To make the learning more stable, I apply the log-transform over the highly variable descriptors.

Subgraph Construction. For each kept residue, a local subgraph is constructed by selecting all residues whose alpha-carbons lie inside the sphere of radius 12 Å, centered around the alpha-carbon of a given residue. To reduce computational complexity, the edges of the subgraph are only created if the distance between the alpha-carbon of two residues is less than 8 Å. Each node of the subgraph contains both the feature vectors and the positional information. Coordinates of alpha-carbons are centered around the analyzed residue to ensure translational invariance.

Geometric Features. The local subgraphs now contain information about residues’ properties and local environment. However, they do not account for residue position with respect to the global protein structure. Instead of computationally intensive tools, such as accessible surface, I use a light approximation. To solve this issue, a vector is constructed from the geometric center of the protein to the alpha carbon of the residue. The length of this vector is normalized by the overall maximum. Furthermore, a plane orthogonal to this vector is defined at its residue end, and the fraction of residues lying further away from this plane is computed. Following similar logic, fractions of residues lying within cones of different angles are computed. This coarse but cheap calculation is aimed at describing the relative position of the subgraph components with respect to the complete protein graph.

Architecture The model (see Figure 8) follows the standard approach for graph neural networks. Firstly, the embeddings of individual nodes are passed through a multi-layer perceptron (MLP) to obtain latent representations. The latter are then transformed through a number of equivariant graph convolution layers (see below). A boolean mask is applied to extract the final representation of a central residue. This representation is concatenated with a vector of global protein features and passed through a dense MLP head to output a single scalar logit. Model architecture is thus kept intentionally simple. Only standard regularization techniques are used, such as dropout and batch normalization.


## Hyperparameters Search

Coarse Search. The first step of the experiment is a coarse hyperparameter search. Due to the computational limitations, the search is performed in two stages rather than via an exhaustive grid search. At first, I check hyperparameters of the model architecture while keeping the remaining parameters fixed to standard values (dropout rate 0.1, binary cross-entropy loss function with positive weight 10, and Adam optimizer with learning rate 0.001).

The model is constructed from three main parts: the encoder, the GNN convolutional block, and the dense prediction head. For each component, the following hyperparameters are investigated: widths of the hidden layers (either 16 or 256); and depths (either 1 or 3 layers). Although I restrict myself to only two options per parameter, it already results in 64 combinations and requires several days of computation. To avoid a possible data leakage, the hyperparameter search is conducted using a separate subset of the dataset, which is not used further for cross-validation. The aim of this coarse search step is to estimate the impact of major architectural choices for every component.

The results are summarized in Figure 9, Left. The F1-score is chosen as a primary metric. On the one hand, the F1-score is suitable for the imbalanced dataset as it reflects a trade- off between false positives and false negatives. Another advantage of using F1 is that it correlates with the commonly used Jaccard score (intersection over union / IOU score).

The visualization requires several clarifications. Both widths and depths are represented as triplets of integer values, corresponding to the encoder block, GNN block, and dense head block, respectively. During training, an early stopping mechanism is used, and the plotted metric is the best validation performance. By comparing combinations pairwise along columns and in rows, one can observe the following:
The encoder with narrow and deep architecture performs consistently poorly (upper- right corner).

Deep GNNs generally outperform shallow ones (darker vertical bands). Increasing the depth of the dense head does not lead to consistent advantage. There are no consistent patterns with respect to layer widths.
There is one combination that outperforms all the others: Encoder (16 x 1), GNN (256 x 3), Head (256 x 3).

Figure 9 (right) gives an alternative perspective on the search results. Although noisy, the figure suggests that the performance has a weak positive correlation with complexity. However, the improvement in performance appears to be approximately linear, while the complexity increases exponentially. It means that investigating more complex models while already being close to the computational limit is excessive due to diminishing returns.

Fine Search. For the next round of hyperparameter search, I take the best architecture found (Encoder (16 x 1), GNN (256 x 3), Head (256 x 3)) and focus on the training- related parameters. The new search grid includes three activation functions: LeakyReLU, ReLU, and ELU; two dropout rates: 0.1 and 0.2; two learning rates: 0.001 and 0.0001; and three positive class weights: 6, 8, and 10.

Selected results are shown in Figure 10. Only runs with positive weight; 10 are presented in the figure, since other positive weights appeared to be less stable during training and inferior in performance. The best validation score is achieved by combining ELU, a dropout rate of 0.1, and a learning rate of 0.001. However, several combinations are com- parable in terms of performance while exhibiting more stable training curves. Especially LeakyReLU with a dropout rate of 0.2 and a learning rate of 0.0001 converges smoothly within just 12 epochs. Note that direct comparison of validation losses is not possible as long as different positive class weights are used in the experiments.


## Cross-Validation and Logits Collection

Cross-Validation. After selecting the hyperparameters, I proceed with evaluating the model performance on a larger dataset using cross-validation. Initially, I intended to perform a 10-fold cross-validation scheme that is widely used and considered particularly suitable for the task [35]. However, the slow training process of GNN and substantial GPU memory requirements, which lead to smaller batch sizes, made me reduce the number of folds. The initial ten subsets of data that are prepared for 10-fold cross-validation are merged pairwise, resulting in 5 subsets. Then a 5-fold cross-validation is performed. Although the variance of the performance estimation becomes higher, this should not significantly affect the objectives of the research.

The average training curves are shown in Figure 11. For each fold, predictions are gen- erated for the test set, producing logits for each residue-centered subgraph. The design of the pipeline keeps track of the protein each particular subgraph originates from. This allows us to aggregate predictions at a protein level.
Logits Collection. Logits are collected from all folds and merged together, which allows us to evaluate the performance of the model for the whole dataset. Although the number of subgraphs exceeds three million, the number of individual proteins is much smaller (9452). From a biological perspective, it makes more sense to compute metrics at the protein level rather than at the residue level, since proteins are the primary units of analysis and goals for prediction.

Using just a single retained test set for evaluation would result in only approximately 800 proteins, which would increase the risk of certain groups being underrepresented and reduce the strength of statistical analysis. The cross-validation, in turn, allows us to collect predictions that cover the whole dataset.

## Residue Level Statistics
As a preliminary step, the residue-level metrics are computed (see Table 4). It can already give a hint of main trends: the difference between structural groups (SCOPe) is substan- tial: from 0.123 F-score for class ’h’ (coiled coils) to 0.536 F1-score for class ’b’ (β-sheet proteins). It is also noticeable that overrepresented classes do not have apparent perfor- mance boosts over underrepresented classes: class ’c’ (α/β) performance is comparable with class ’j’ (peptides) performance, while class ’c’ has almost 1500 times more samples. Another preliminary observation is that functional classes perform more consistently and also benefit from broader representation: the best-performing class, ’2’ (transferases), is also the most abundant, while the worst-performing class, ’7’ (translocases), has the least number of samples. For the kingdom of the source organism, the preliminary results are curious: viral proteins have dominating performance, bacterial and archean proteins have almost indistinguishable metrics, while eukaryote proteins performance lies in-between.

## Protein Level Statistics
The next step is to compute the metrics at the protein level. These metrics lead to more sensible and interpretable results and also allow us to conduct statistical analysis. In Figure 12, the F1-scores of the protein groups according to different classifications are shown as a combination of a box plot (with standard definitions for body, whiskers, and outliers) and a violin plot. Notice that box plots for the particular ligands and organisms are located in Appendix C (Figure 14, Figure 15). The box plot alone does not fully describe distribution shape but is easier for immediate comparison and interpretation.

The violin plot in the background of a box plot augments the information with the shape of the distribution and can show clusters, which are not identified as outliers by the box plot. The numbers on top of each group in the plots indicate the number of proteins in the dataset attributed to this group.

Visually, there are SCOPe classes with clearly worse performance: class ’f’ (membrane proteins) and class ’g’ (small proteins). The leader by performance is class ’b’ (β-sheet proteins). For the EC classification, the difference is not drastic, with class ’2’ (trans- ferases) performing slightly better than other enzymatic groups. Within protein groups by size, the discrepancy is prominent: the performance is the highest for medium-sized protein with 100-200 residues, while too short or too long proteins tend to have poorer F1-scores, with those having more than 1000 residues featuring almost three times drop in scores compared to the leading group. If we group the proteins according to the kingdom of its source organism, the viral proteins demonstrate the highest F1-scores followed by the eukaryotic proteins. Archean and bacterial proteins exhibit the lowest performance with almost identical results. The Figure 14 shows the results by different organisms (only for those which are represented in the dataset with at least 70 samples). Some ob- vious problems are observable on this plot: the distribution of hepatitis C virus proteins has two distinct peaks, HIV-related proteins show suspiciously high results compared to other source organisms.

A closer inspection of the dataset and logits reveals a cluster of proteins with a mean F1- score of 0.846. As shown in Table 5, the HIV-related proteins have substantial differences from the overall dataset. These proteins are typically shorter, less structurally diverse, and have more distinct UniProt-ligand pairs for particular proteins (up to 66). Additionally, more than 80% of enzymes belong to the transferases class compared to 40% in the full dataset. This pattern also holds for virus-related samples in general, where the fraction of transferases is around 80%.

These observations indicate the presence of a relatively homogeneous subgroup of struc- turally and functionally similar proteins (254) that feature disproportionately strong predictions compared to the rest of the dataset. The impact of this subgroup is also visible in several distributions: the violin plots in Figure 12 (a) (second density peak at beta-sheets); Figure 12 (b) (density peak of transferases is skewed); Figure 12 (c) (second peak for proteins with lengths 100-200); Figure 12 (d) (second peak for viruses). Thus, this highly homogeneous group with disproportionately high performance is excluded from further analysis to avoid the potential bias. The new box plots after the cluster removal can be seen in Appendix C.

## Statistical Tests
Kruskal-Wallis Statistics. For statistical analysis, a Kruskal-Wallis test [38] is applied, as there are several classifications with more than two groups in each of them. The distribution is not supposed to be normal, so the rank-based test is chosen. The Kruskal- Wallis is estimated for different classifications, and if the p-value is found to be significant (below .05), which means that at least one group does not originate from the same distribution as others, post-hoc analysis is conducted for pairs of groups. For pairwise analyses, Dunn’s post-hoc test [16] is selected. Dunn’s test includes testing correction (Holm-Bonferroni adjustment is chosen) and is suitable for multiple comparisons.

The combined results of the Kruskal-Wallis test are presented in (Table 6). For robustness, only groups with enough samples are taken into account in each classification. It is also worth noting that in the case of size classification, the p-value is smaller than the 64- bit floating-point data type can handle. It is immediately visible that all the tests are significant at both the .95 and 0.99 levels. Thus, a post-hoc pairwise analysis is conducted for every classification.

Post-hoc Dunn’s test. The results of Dunn’s tests are represented as heatmaps in Fig- ure 13. Dunn’s test heatmaps for proteins grouped by organism and ligand are located in the Appendix C. The visual analysis reveals that the pairwise test is significant for the majority of pairs of SCOPe classes, although membrane proteins, multi-domain proteins, and small proteins do not show significant differences. The EC classification unveils a curious pattern: hydrolases and transferases are significantly different in terms of per- formance from all other enzyme classes and also from each other. The performance of oxidoreductases is also noticeably different from that of the majority of other functional classes. The p-values heatmap of size classification suggests that small proteins (less than 100 residues) do not perform significantly differently compared to medium-sized proteins (100-400 residues). Almost all the other comparisons feature p-values below .05 or below .01. The situation with proteins grouped according to the kingdom of their source organ- ism implies that eukaryotic proteins exhibit significantly different performance against other kingdoms; however, pairwise comparisons between bacterial, archean, and viral pro- teins have p-values high above the threshold. Additionally, comparisons between vastly represented organisms indicate the strong particularity of human-derived proteins (see Figure 19). In terms of classification by frequent ligands, proteins associated with coen- zyme A (RCSB ID: "COA") or NADP (nicotinamide adenine dinucleotide phosphate, RCSB ID: "NAP") have F1-scores significantly different from other ligands (see Fig- ure 20).

## Effect Sizes and Latent Scores
Cliff’s Deltas. The Dunn’s test only gives a hint about potentially valuable comparisons. To determine the effect of different groups on the model performance, I rely on effect size measure via Cliff’s Delta [12]:
PP
δ = i,j 1(xiA > xjB)− i,j 1(xiA < xjB) (8) nA nB
where xiA - the i-th element of group A; xjB - the j-th element of group B; n1,n2 - the sizes of the groups; and 1(f) - a predicate returning 1 if condition f is true and 0 otherwise. The order of the groups in a pair matters.
The methodology is the following: for every comparison that has a p-value below .05 according to the Dunn’s test, the Cliff’s delta is computed. The positive value of delta indicates the higher performance of group A, while the negative delta indicates the higher performance of group B. The absolute value of the delta can be interpreted as an "effect size". In the scope of this research, it can be viewed as a tendency towards better per- formance in group A compared to group B. For this research, I consider the magnitude below 0.15 negligible; from 0.15 up to 0.33 as small; from 0.33 up to 0.47 as medium; from 0.47 up to 0.6 as large; and above 0.6 as very large. The summary of all pairs of groups that are simultaneously significant according to Dunn’s test and have a non-negligible Cliff’s delta is shown in Table 7.
Latent Scores from Linear Model. After taking a closer look at the significant pair- wise comparisons, one can notice that there are no cycles in the results, which suggests that pairwise comparisons are approximately consistent with global ordering (transitive assumption). This observation, in turn, allows me to construct a one-dimensional latent rating score by learning latent factors via linear regression. Therefore, a latent group ranking can be estimated by fitting the model using difference pairs and delta values. Let G = {1,...,Ngroups} be the set of groups within classification. For each significant pairwise comparison i = 1, . . . , n, let ai , bi ∈ G stand for the groups compared, and let di be the corresponding Cliff’s delta for the comparison ai versus bi (comparison is ordered). A matrix X ∈ Rn×Ngroups is constructed as
 1,
Xij = −1, 
0,
j = ai,
j = bi, (9) otherwise.
The effect sizes are then represented as the differences between latent scores. Including the noise:
di =wai −wbi +εi. (10) This can equivalently be formulated in matrix notation:
d = Xw + ε, (11)
where w ∈ RNgroups is a vector that contains the latent scores of the groups. The well-known least-squares estimate is therefore
Finally, since the data only include differences, the latent scores vector can only be defined
up to an additive constant. For comparability reasons, scores are centered after fitting:
Ngroups X
The coefficient of determination (R2) of the fitted linear models remains above 0.97 (mostly above 0.99) across all the classifications used, which supports the assumption that the pairwise deltas are well explained by a compact one-dimensional ordering.

The final result of the research is shown in Table 9. The similar table for species- based and ligand-based classifications can be found in Appendix D as Table 27. In the table, the centered latent scores and F1 scores are summarized. There are also win- lose integer scores that are evaluated as the number of wins over other groups through significant comparisons minus the number of losses. For the sake of completeness, the group sizes are also added to the table. The consistency of values in the table strengthens the results: the latent rating scores, the F1 scores, and win/lose scores are in agreement in terms of ranking in each of the four major classifications.

## Conclusion and Limitations
Grouping by Size. The initial hypothesis that the GNN-based model exhibits different performance across different structural, functional, and source-based protein groups is supported by experimental results. Structural groups, which include a naive classification by protein sequence length and a more robust classification according to SCOPe, show the largest variation in performance metrics.

Interestingly, size-based grouping appears to have a larger impact on F1 scores than structural motifs. Based on the findings, two groups can be suggested. The first group includes proteins with up to 400 residues; the second group contains proteins with more than 400 residues. The proteins of the first group achieve an average F1 score of 0.522 and tend to perform better compared to the dataset in general. The proteins in the second group exhibit an average F1 score of 0.314, which is worse than average for the full dataset.

A more refined analysis based on the linear regression on Cliff’s deltas reveals further details. The best-performing subset includes proteins with 200-300 residues, followed by those composed of 100-200 residues with a close performance. Thus, proteins with 100- 300 residues are the most favorable for efficient binding site prediction using the tested GNN architecture. Both shorter and longer proteins show reduced performance. Proteins longer than 400 residues show a progressive decrease in performance with additional length. The poorest performance is observed for the group ’>1000 residues’, which is significantly worse in terms of F1-score than every other group, according to the Dunn’s post-hoc test (on significance level .95).

There might be several reasons for such results:
Dataset bias: proteins with 100-400 residues are more abundant in the dataset;
The GNN uses subgraphs of fixed radius; large proteins might have long-reach effects that are not captured by subgraphs in the case of larger proteins;
Larger proteins are often multi-domain with more diverse local environments;
Too small proteins (<100 residues) might lack a stable spatial conformation or are not reliably resolved.

Grouping by Structure. The SCOPe classification shows less pronounced effect sizes but still reveals interesting patterns. The SCOPe classes ’e’ (multi-domain proteins), ’f’ (membrane proteins), and ’g’ (small proteins) are consistently inferior in performance to the rest of the classes. In the case of multi-domain and small proteins, the reason behind it might be structural complexity. The membrane proteins exist in an environment that is different from the proteins in the solution. They typically have a hydrophobic belt and polar surfaces that create specific local properties. Another reason for poor performance might be the smaller number of samples.
Among higher-performing groups, α + β (segregated) achieves the highest average F1- score of 0.495 and a latent score of 0.314, which is higher than for α-helices or for β-sheets alone. This might be attributed to the abundance of samples or conservative motifs, such as the ferredoxin fold. Interestingly, the α/β(interleaved) class performs significantly weaker than the α + β(segregated) despite having more samples, suggesting that the model struggles to capture alternating secondary structures. Another result is a slight performance advantage of β-only proteins compared to the α-only proteins. However, the pairwise comparison between them is not statistically significant. A class of structural artifacts (’l’) shows the second highest performance. The reason for it is not clear and might require more in-depth analysis.

Grouping by Function. Performance across functional classes is more uniform than across structural classes. The F1-score of the best-performing class (transferases) is only 31% higher than the average score of the worst-performing class (isomerases). Another observation in favor of a limited importance of functional classes for the binding site prediction is that the F1-scores tend to decrease for classes with fewer available samples. In the pairwise comparisons, transferases and hydrolases show a moderate positive effect over other (underrepresented) classes. However, this observation should be interpreted cautiously, since more than half of the transferases in the dataset belong to a single subclass (2.7).

Grouping by Kingdom. The classification by kingdom of source organism yields less consistent results. Eukaryotic proteins exhibit moderately high F1 scores compared to three other kingdoms, while bacterial, archaeal, and viral proteins are not statistically distinguishable. Although the gap in effect size between eukaryotic proteins and other groups is not large, the average F1-scores differ a lot (0.465 vs. 0.391). This gap might be partially explained by the higher number of eukaryotic proteins in the dataset. Fur- thermore, 49.3% of the bacterial proteins in the dataset belong to the SCOPe class ’c’ (α/β), while only 26.5% of the eukaryotic proteins belong to this class. The fraction of α + β proteins, on the contrary, is 36.3% of eukaryotic proteins and only 25.4% of bacterial proteins that belong to this class. Provided that the effect sizes of structural groups are higher than that of the kingdoms, the gap is likely explained by the structural distribution rather than evolutionary factors.

Grouping by Species. Additional classifications also reveal several valuable observa- tions. The comparison of performance per species (limited to the species with at least 50 proteins in the dataset) shows the dominance of Homo sapiens, which is by a significant margin the most frequent source species in the dataset. However, the performance of pro- teins originating from Mus musculus, Bos taurus, and Rattus norvegicus is only 10-20% worse than that of Homo sapiens despite 15-20 times fewer samples. The only signifi- cant pairwise comparisons that involve organisms of the same kingdom (if we omit the overrepresented Homo sapiens) are Oryctolagus cuniculus (76 samples) vs. Mus musculus (208 samples) and Oryctolagus cuniculus vs. Bos taurus (137 samples). This evidence is not enough to confirm that the species the protein is extracted from has a significant impact on the quality of binding site predictions.

Grouping by Ligand. The ligand-based classification reveals that all significant com- parisons involve coenzyme A or NADP. Proteins resolved with coenzyme A show sig- nificantly lower performance than almost any other ligand-protein complex. Proteins resolved with NADP show the inverse pattern: their performance is significantly higher than that of proteins associated with several other ligands. Neither of these special cases is overrepresented in the dataset. One possible explanation for the poor performance of coenzyme A-associated proteins is that coenzyme A is a large ligand (> 700 Da) with 18 rotatable bonds, which might lead to diverse binding sites compared to conserva- tive sites for NAD, NADP, FAD, ATP, etc. This suggests that a binding site prediction performance depends on the ligand structural properties.

Summarizing the findings:
- Protein size has the strongest observed effect on the model performance, with proteins containing 100-400 residues performing best;
- The model benefits from spatially separated α-helices and β-sheets;
- Membrane, multi-domain, and small proteins are harder for a model to predict;
- Functional classes have a relatively small effect on the model performance;
- The source of the protein (species, kingdom) appears to be weak and coupled with structural properties;
- The chemical properties of the ligand might have an effect on the model performance; however, further research is required with more samples per different types of ligands.

Further Research. These findings alone are not strictly conclusive; however, they might serve as a basis for further investigation and more detailed hypotheses. Among obvious continuations: the cross-classification analysis to understand how spatial and functional subsets are interconnected in terms of effect on performance. Another problem that has already been mentioned in earlier sections includes a cluster of HIV-related proteins. Considering the finding of a shifted structural distribution of bacterial proteins in the dataset, it is potentially helpful to train different models for eukaryotic, bacterial, and vi- ral proteins and compare the performance with a single model. Other suggestions include using separate preprocessing approaches for short and long proteins, such as extending the subgraph radius for large proteins. Finally, the SCOPe classification analysis reveals that the model trained on frequent motifs struggles with special cases such as membrane proteins or small proteins lacking stable secondary structure. There are a limited number of resolved structures of these classes, which can hinder the training of a potent model. However, it is worth trying to use separate dense heads for different SCOPe classes to combine robust encoding with a class-aware decision process.

Limitations. The findings of this work are subject to several limitations. First, the last official release of the scPDB database dates back to 2017 and only contains around 16,000 samples, of which only 9,452 are used for this research, which is relatively few compared to the complete Protein Data Bank. Second, data leakage may occur if certain proteins are structurally similar, which cannot be completely avoided by UniProt-based split alone. Third, due to computational limitations, the hyperparameter search for the model is limited. Finally, it cannot be guaranteed that the relative performance across groups remains invariant under different model configurations.

Overall, the results demonstrate that the binding site prediction performance using a GNN-based model is substantially different across size-based and structural groups, while functional and evolutionary groups are found to have weaker influence. This suggests the importance of considering the structural diversity of proteins while designing and evaluat- ing models for the task. Although subject to computational and data-driven limitations, these conclusions may contribute to further model-design strategies and the development of group-specific approaches.


