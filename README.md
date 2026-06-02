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
Proteins are large complex molecules composed of sequences of 20 different amino acids, which are arranged in one or more polypeptide chains. Drugs are typically small molecules that are capable of producing biological effects when delivered to the organism. An interaction between a drug (ligand) and a protein (target) is usually non-covalent and occurs in a specific region of a protein called a ’binding site’ or a ’binding pocket’. Thus, identifying binding sites helps to understand the way a protein functions, the properties of potential ligands, and is an essential step for efficient drug design [56].

<img width="4000" height="3000" alt="1dmp" src="https://github.com/user-attachments/assets/4b8c914d-8596-4dcc-a202-31692f0e5be6" />
Figure 1. HIV-1 protease (blue) complex with DMQ/Mozenavir (yel-
low) resolved in the pocket (red). PDB: 1dmp.

<img width="3172" height="2356" alt="drug_discovery" src="https://github.com/user-attachments/assets/68aab60a-b4d8-4929-8a30-b3018f234282" />
Figure 2. Drug Discovery pipeline [32], research-relevant steps highlighted with gray background.

The three-dimensional structures of proteins form intricate surfaces with numerous concavities and protrusions, creating distinct microenvironments for ligand binding and catalysis [48]. Therefore, binding sites are typically localized regions composed of several residues. These regions are particularly favorable for recognition and interaction with specific ligands.

There exist several paradigms in drug discovery[83]. Structure-based drug discovery (SBDD) leverages knowledge of target molecules - usually proteins - to select or design possible ligands that would interact with a target [32]. The initial steps of a classical SBDD include identifying the target protein, determining po- tentially druggable binding pockets, performing virtual screening and docking algorithms (in silico), and detecting candidates [83] (see Figure 1, Right).

The main goal of this research is to assess the comparative performance of binding- site prediction across different groups of proteins and to suggest future work based on the findings.

## Materials

| Metric | Value |
|----------|----------|
| Total Entries | 17,594 |
| Unique PDB IDs | 16,612 |
| Unique UniProt IDs | 4,688 |
| Mean Resolution (Å) | 2.14 |
| Unique Species | 927 |
| Most Common Species | *H. sapiens*, *E. coli*, *R. norvegicus* |
| Most Common Kingdoms | Eukaryota, Bacteria, Viruses |
| SCOPe Classes Present | 776 |
| Most Common SCOPe Classes | d.144.1, l.1.1, b.50.1 |
| Binding Sites with Metals | 3,323 |
| Common Metals | Mg, Zn, Mn |
| Binding Sites with Cofactors | 1,389 |
| Common Cofactors | NAP, FAD, NDP |
| Mean Binding Site Size | 36.45 residues |
| Mean Standard Residues | 34.49 |
| Mean Cavity Volume (Å³) | 792.66 |
| Mean Hydrophobic Fraction (%) | 49.06 |
| Mean Polar Fraction (%) | 50.94 |
Table 1. Selected aggregated values of scPDB (v.2017)

The primary source of protein structures and related information for the study is scPDB [15]. scPDB is an annotated database of druggable binding sites derived from the Protein Data Bank (PDB) [4]. The scPDB mainly includes information on small synthetic and natural ligands, along with corresponding high-quality, non- redundant protein binding sites. For more structural information and for later analyses, I also made use of the latest stable release of a SCOPe database [11]: a database that classifies protein chains according to their structure, function, and taxonomy.

By March 15th, 2026, there are 17,594 entries available for download from the official page of scPDB. However, according to the information on the official site, the last stable release (2017) contains 16,034 entries, 4,782 proteins, and 6,326 ligands. In order to collect more robust metadata and get potential insights into data patterns, I have downloaded the scPDB web pages of as many entries as are available. Using the BeautifulSoup4 module [55], I have successfully found 16,032 (91.1% of the database) entries, which is only two less than a stable release contains. Apart from scPDB, I have gathered additional descriptors through RCSB API[57], which provides various information about PDB entries, including citations, refinements, experimental setup, distinct molecules, aggregate metrics, etc. The main API endpoint is available for 17,539 entries (99.7%).

[scopes.pdf](https://github.com/user-attachments/files/28502932/scopes.pdf)

Figure 3. Visual representation of SCOPe subclasses present in the dataset.
Indicates the dominance of certain structural patterns and a substantial portion of artifacts (l.1).

[chain_config.pdf](https://github.com/user-attachments/files/28502970/chain_config.pdf)
Figure 4. Chain Configuration of proteins in the dataset. The numbers on the heatmap stand for the count of entries. The majority of entries fall into a bottom-left region.

There are only 4688 unique UniProt identifiers, 903 PDB codes are encountered twice. The lengths of peptide chains are highly variable. The values span from only 35 residues to several thousand residues (according to mol2 file analysis). The chain composition tends to be simple in most cases. 87.6% of entries can be described with a single FASTA sequence (although the number of identical monomers varies). 9.18% have 2 unlike chain types, 1.89% have 3 unlike chain types, and 1.33% have more than 3 (up to 14). 
Considering the function, for 56.3% of proteins in the dataset, enzyme codes are known. The most frequent classes are transferases (19.9%), oxyreductases (13%), and hydrolases (9.67%). 

The further analysis of the data gives a better view of the construction and properties of the binding sites. The average cavity volume is estimated to be 793, while the average volume of a cubic grid around a protein is 313,000.  The mean number of binding residues per protein is 33.6 (SD 11.9).

Only distinct protein structures are kept.  In total, 9452 entries are therefore left. The samples are split into 6 portions with UniProt IDs distributed without intersections to prevent data leakage. A 5-fold cross-validation is performed while 1 portion is retained for hyperparameter search. Cross-validation is known to be particularly suitable when a disjoint validation set is not feasible.

## Methodology

The following design choices are therefore imposed:

1. Features passed to the model are extracted on the residue level.
2. Precomputed features.
3. Each residue is represented by a set of predefined features, including length (Å), width (Å), volume (Å3), pKa constant, charge, hydropathy, etc.
4. For each kept residue, a local subgraph is constructed by selecting all residues whose alpha-carbons lie inside the sphere of radius 12 Å, centered around the alpha-carbon of a given residue. 
5. A vector is constructed from the geometric center of the protein to the alpha carbon of the residue to account for overall position.
6. The model follows the standard approach for graph neural networks.

[model.pdf](https://github.com/user-attachments/files/28503026/model.pdf)

Figure 5. Sketch of the model architecture. Green boxes are part of the input.
The blue box is the output.

## Hyperparameters Search

At first, I check hyperparameters of the model architecture while keeping the remaining parameters fixed to standard values (dropout rate 0.1, binary cross-entropy loss function with positive weight 10, and Adam optimizer with learning rate 0.001).

[hp_broad.pdf](https://github.com/user-attachments/files/28503045/hp_broad.pdf)
Figure 6. Summary of the broad hyperparameter search. In every triplet of values, the first number refers to the encoder, the second to the GNN, and the third to the dense head.

By comparing combinations pairwise along columns and in rows, one can observe the following:
The encoder with narrow and deep architecture performs consistently poorly (upper- right corner).Deep GNNs generally outperform shallow ones (darker vertical bands). Increasing the depth of the dense head does not lead to consistent advantage. There are no consistent patterns with respect to layer widths. There is one combination that outperforms all the others: Encoder (16 x 1), GNN (256 x 3), Head (256 x 3).

[hp_complexity.pdf](https://github.com/user-attachments/files/28503099/hp_complexity.pdf)

Figure 7. F1-score plotted against the complexity of the model (in log-scale). A linear model is fitted on the data after log-transforming
the complexity.

For the next round of hyperparameter search, I take the best architecture found (Encoder (16 x 1), GNN (256 x 3), Head (256 x 3)) and focus on the training-related parameters. The new search grid includes three activation functions: LeakyReLU, ReLU, and ELU; two dropout rates: 0.1 and 0.2; two learning rates: 0.001 and 0.0001; and three positive class weights: 6, 8, and 10.

<img width="1189" height="1180" alt="hp_grid" src="https://github.com/user-attachments/assets/07b5f8fa-c4d9-40e0-9ee1-38c73744407c" />

Figure 8. Validation F1-score by epoch for several runs during hyperparameter search. Values in the subplot titles refer to the activation function / dropout / learning rate / positive weight. Dashed lines indicate the best metric per run and its epoch, while red lines indicate the best overall combinations.

The best validation score is achieved by combining ELU, a dropout rate of 0.1, and a learning rate of 0.001. However, several combinations are com- parable in terms of performance while exhibiting more stable training curves. Especially LeakyReLU with a dropout rate of 0.2 and a learning rate of 0.0001 converges smoothly within just 12 epochs. Note that direct comparison of validation losses is not possible as long as different positive class weights are used in the experiments.


## Cross-Validation and Logits Collection

After selecting the hyperparameters, I proceed with evaluating the model performance on a larger dataset using cross-validation. For each fold, predictions are generated for the test set, producing logits for each residue-centered subgraph. Logits are collected from all folds and merged together, which allows us to evaluate the performance of the model for the whole dataset.

## Residue Level Statistics

<img width="686" height="682" alt="Screenshot 2026-06-02 at 11 52 52" src="https://github.com/user-attachments/assets/82a3cb32-fda8-4a2e-8c22-49ef4f667f60" />
Table 2. Bulk aggregated metrics of different protein groups on residues level. The highest values - in bold,the second highest - in italics

As a preliminary step, the residue-level metrics are computed (bulk). The difference between structural groups (SCOPe) is substantial: from 0.123 F-score for class ’h’ (coiled coils) to 0.536 F1-score for class ’b’ (β-sheet proteins). It is also noticeable that overrepresented classes do not have apparent performance boosts over underrepresented classes: class ’c’ (α/β) performance is comparable with class ’j’ (peptides) performance, while class ’c’ has almost 1500 times more samples.

## Protein Level Statistics

<img width="966" height="718" alt="Screenshot 2026-06-02 at 11 53 15" src="https://github.com/user-attachments/assets/6a0b0bde-d8bb-4276-8f33-6713ba571411" />
Figure 8. Comparison of model performance (F1-scores) across different classifications (numbers on top indicate the number of proteins per class). Top Left: SCOPe; Top Right: EC; Bottom Left: Size; Bottom Right: Kingdom.

The next step is to compute the metrics at the protein level. These metrics lead to more sensible and interpretable results and also allow us to conduct statistical analysis.

Visually, there are SCOPe classes with clearly worse performance: class ’f’ (membrane proteins) and class ’g’ (small proteins). The leader by performance is class ’b’ (β-sheet proteins). For the EC classification, the difference is not drastic, with class ’2’ (trans- ferases) performing slightly better than other enzymatic groups. Within protein groups by size, the discrepancy is prominent: the performance is the highest for medium-sized protein with 100-200 residues, while too short or too long proteins tend to have poorer F1-scores, with those having more than 1000 residues featuring almost three times drop in scores compared to the leading group. If we group the proteins according to the kingdom of its source organism, the viral proteins demonstrate the highest F1-scores followed by the eukaryotic proteins. Archean and bacterial proteins exhibit the lowest performance with almost identical results. The Figure 14 shows the results by different organisms (only for those which are represented in the dataset with at least 70 samples). Some ob- vious problems are observable on this plot: the distribution of hepatitis C virus proteins has two distinct peaks, HIV-related proteins show suspiciously high results compared to other source organisms.

## Statistical Tests
For statistical analysis, a Kruskal-Wallis test [38] is applied, as there are several classifications with more than two groups in each of them. It is immediately visible that all the tests are significant at both the .95 and 0.99 levels. Thus, a post-hoc pairwise analysis is conducted for every classification. The results of post-hoc Dunn’s tests are represented as heatmaps. The visual analysis reveals that the pairwise test is significant for the majority of pairs of SCOPe classes, although membrane proteins, multi-domain proteins, and small proteins do not show significant differences. The EC classification unveils a curious pattern: hydrolases and transferases are significantly different in terms of per- formance from all other enzyme classes and also from each other. The performance of oxidoreductases is also noticeably different from that of the majority of other functional classes. The p-values heatmap of size classification suggests that small proteins (less than 100 residues) do not perform significantly differently compared to medium-sized proteins (100-400 residues). Almost all the other comparisons feature p-values below .05 or below .01.

<img width="807" height="710" alt="Screenshot 2026-06-02 at 11 53 57" src="https://github.com/user-attachments/assets/58b04c85-06e4-4a9c-ad01-dda8ba7bdfcd" />
Figure 9. Dunn’s test results for different classifications. P-values above .05 are colored with shades of red; values below .05 are colored with shades of blue. Negative integers mean decimal logs of small p-values. Top Left: SCOPe; Top Right: EC; Bottom Left: Size; Bottom Right: Kingdom

## Effect Sizes and Latent Scores

### Cliff's Delta

To quantify the magnitude and direction of differences between protein groups, I use **Cliff's Delta**:

$$
\delta =
\frac{
\sum_{i,j}\mathbf{1}(x_{iA}>x_{jB})
-
\sum_{i,j}\mathbf{1}(x_{iA}<x_{jB})
}
{n_A n_B}
$$

where:

- $begin:math:text$x\_\{iA\}$end:math:text$ – i-th observation from group A
- $begin:math:text$x\_\{jB\}$end:math:text$ – j-th observation from group B
- $begin:math:text$n\_A\, n\_B$end:math:text$ – group sizes
- $begin:math:text$\\mathbf\{1\}\(\\cdot\)$end:math:text$ – indicator function

Interpretation:

| $begin:math:text$\|\\delta\|$end:math:text$ | Effect Size |
|-------------|-------------|
| < 0.15 | Negligible |
| 0.15 – 0.33 | Small |
| 0.33 – 0.47 | Medium |
| 0.47 – 0.60 | Large |
| > 0.60 | Very Large |

Only pairwise comparisons that were significant according to Dunn's post-hoc test ($begin:math:text$p \< 0\.05$end:math:text$) were considered.

---

### Latent Group Ranking

Significant pairwise Cliff's deltas were converted into a one-dimensional latent ranking.

For each significant comparison between groups $begin:math:text$a\_i$end:math:text$ and $begin:math:text$b\_i$end:math:text$, a row of the design matrix is defined as

$$
X_{ij}=
\begin{cases}
1,& j=a_i \\
-1,& j=b_i \\
0,& \text{otherwise}
\end{cases}
$$

The model assumes

$$
d_i = w_{a_i}-w_{b_i}+\varepsilon_i,
$$

or equivalently

$$
\mathbf d = \mathbf X \mathbf w + \boldsymbol\varepsilon.
$$

The latent scores are estimated using ordinary least squares:

$$
\hat{\mathbf w}
=
(\mathbf X^T\mathbf X)^{-1}
\mathbf X^T
\mathbf d.
$$

Scores are centered after fitting because only pairwise differences are identifiable.

The resulting models achieved $begin:math:text$R\^2 \> 0\.97$end:math:text$ for all investigated classifications, indicating that a one-dimensional latent ordering explains the observed effect sizes remarkably well.

<img width="596" height="679" alt="Screenshot 2026-06-02 at 11 54 23" src="https://github.com/user-attachments/assets/da59ca37-054e-43f9-996b-79425a53199e" />
Table 3. Final result for classifications by SCOPe, EC, Size and Reign (Kingdom). (Sorted by Latent Factor from high to low)

## Conclusion and Limitations
Grouping by Size. The initial hypothesis that the GNN-based model exhibits different performance across different structural, functional, and source-based protein groups is supported by experimental results. Structural groups, which include a naive classification by protein sequence length and a more robust classification according to SCOPe, show the largest variation in performance metrics.

A more refined analysis based on the linear regression on Cliff’s deltas reveals further details. The best-performing subset includes proteins with 200-300 residues, followed by those composed of 100-200 residues with a close performance. Thus, proteins with 100- 300 residues are the most favorable for efficient binding site prediction using the tested GNN architecture. Both shorter and longer proteins show reduced performance. Proteins longer than 400 residues show a progressive decrease in performance with additional length. The poorest performance is observed for the group ’>1000 residues’, which is significantly worse in terms of F1-score than every other group, according to the Dunn’s post-hoc test (on significance level .95).

Grouping by Structure. The SCOPe classification shows less pronounced effect sizes but still reveals interesting patterns. The SCOPe classes ’e’ (multi-domain proteins), ’f’ (membrane proteins), and ’g’ (small proteins) are consistently inferior in performance to the rest of the classes. In the case of multi-domain and small proteins, the reason behind it might be structural complexity. The membrane proteins exist in an environment that is different from the proteins in the solution. They typically have a hydrophobic belt and polar surfaces that create specific local properties. Another reason for poor performance might be the smaller number of samples.
Among higher-performing groups, α + β (segregated) achieves the highest average F1- score of 0.495 and a latent score of 0.314, which is higher than for α-helices or for β-sheets alone. This might be attributed to the abundance of samples or conservative motifs, such as the ferredoxin fold. Interestingly, the α/β(interleaved) class performs significantly weaker than the α + β(segregated) despite having more samples, suggesting that the model struggles to capture alternating secondary structures. Another result is a slight performance advantage of β-only proteins compared to the α-only proteins. However, the pairwise comparison between them is not statistically significant. A class of structural artifacts (’l’) shows the second highest performance. The reason for it is not clear and might require more in-depth analysis.

Grouping by Function. Performance across functional classes is more uniform than across structural classes. The F1-score of the best-performing class (transferases) is only 31% higher than the average score of the worst-performing class (isomerases). Another observation in favor of a limited importance of functional classes for the binding site prediction is that the F1-scores tend to decrease for classes with fewer available samples. In the pairwise comparisons, transferases and hydrolases show a moderate positive effect over other (underrepresented) classes. However, this observation should be interpreted cautiously, since more than half of the transferases in the dataset belong to a single subclass (2.7).

Grouping by Kingdom. The classification by kingdom of source organism yields less consistent results. Eukaryotic proteins exhibit moderately high F1 scores compared to three other kingdoms, while bacterial, archaeal, and viral proteins are not statistically distinguishable. Although the gap in effect size between eukaryotic proteins and other groups is not large, the average F1-scores differ a lot (0.465 vs. 0.391). This gap might be partially explained by the higher number of eukaryotic proteins in the dataset. Fur- thermore, 49.3% of the bacterial proteins in the dataset belong to the SCOPe class ’c’ (α/β), while only 26.5% of the eukaryotic proteins belong to this class. The fraction of α + β proteins, on the contrary, is 36.3% of eukaryotic proteins and only 25.4% of bacterial proteins that belong to this class. Provided that the effect sizes of structural groups are higher than that of the kingdoms, the gap is likely explained by the structural distribution rather than evolutionary factors.

Summarizing the findings:
- Protein size has the strongest observed effect on the model performance, with proteins containing 100-400 residues performing best;
- The model benefits from spatially separated α-helices and β-sheets;
- Membrane, multi-domain, and small proteins are harder for a model to predict;
- Functional classes have a relatively small effect on the model performance;
- The source of the protein (species, kingdom) appears to be weak and coupled with structural properties;
- The chemical properties of the ligand might have an effect on the model performance; however, further research is required with more samples per different types of ligands.

## References

[1] Rishal Aggarwal, Akash Gupta, Vineeth Chelur, C. V. Jawahar, and U. Deva
Priyakumar. 2021. DeepPocket: Ligand Binding Site Detection and Segmentation
using 3D Convolutional Neural Networks. ChemRxiv, 2021, 0715. doi: 10.26434/ch
emrxiv-2021-7fkkx-v2. eprint: https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-
2021-7fkkx-v2. https://chemrxiv.org/doi/abs/10.26434/chemrxiv-2021-7fkkx-v2.
[2] Ethan C. Alley, Grigory Khimulya, Surojit Biswas, Mohammed AlQuraishi, and
George M. Church. 2019. Unified rational protein engineering with sequence-only
deep representation learning. bioRxiv. doi: 10.1101/589333. eprint: https://www
.biorxiv.org/content/early/2019/03/26/589333.full.pdf. https://www.biorxiv.org
/content/early/2019/03/26/589333.
[3] Christof Angermueller, Tanel Pärnamaa, Leopold Parts, and Oliver Stegle. 2016.
Deep learning for computational biology. en. Molecular Systems Biology, 12, 7,
(July 2016), 878. issn: 1744-4292, 1744-4292. doi: 10.15252/msb.20156651. Re-
trieved 03/28/2025 from https://www.embopress.org/doi/10.15252/msb.2015665
1.
[4] H. M. Berman. 2000. The Protein Data Bank. Nucleic Acids Research, 28, 1, (Jan-
uary 2000), 235–242. issn: 13624962. doi: 10 . 1093 / nar / 28 . 1 . 235. Retrieved
03/28/2025 from https : / / academic . oup . com / nar / article - lookup / doi / 10 . 1093
/nar/28.1.235.
[5] T. Andrew Binkowski, Shapor Naghibzadeh, and Jie Liang. 2003. CASTp: Com-
puted Atlas of Surface Topography of proteins. Nucleic Acids Research, 31, 13,
(July 2003), 3352–3355. issn: 0305-1048. doi: 10 . 1093 / nar / gkg512. eprint: htt
ps : / / academic . oup . com / nar / article - pdf / 31 / 13 / 3352 / 9487117 / gkg512 . pdf.
https://doi.org/10.1093/nar/gkg512.
[6] G P Brady Jr and P F Stouten. 2000. Fast prediction and visualization of protein
binding pockets with PASS. en. J Comput Aided Mol Des, 14, 4, (May 2000), 383–
401.
[7] Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Van-
dergheynst. 2017. Geometric Deep Learning: Going beyond Euclidean data. IEEE
Sig. Proc. Mag., 34, 4, 18–42. doi: 10.1109/MSP.2017.2693418. arXiv: 1611.08097
[cs.CV].
[8] Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. 2014. Spectral
Networks and Locally Connected Networks on Graphs. In 2nd International Con-
ference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16,
2014, Conference Track Proceedings. Yoshua Bengio and Yann LeCun, (Eds.) http
://arxiv.org/abs/1312.6203.
[9] Michal Brylinski and Jeffrey Skolnick. 2007. A threading-based method (FIND-
SITE) for ligand-binding site prediction and functional annotation. en. Proc Natl
Acad Sci U S A, 105, 1, (December 2007), 129–134.
[10] John A. Capra, Roman A. Laskowski, Janet M. Thornton, Mona Singh, and
Thomas A. Funkhouser. 2009. Predicting Protein Ligand Binding Sites by Combin-
ing Evolutionary Sequence Conservation and 3D Structure. PLOS Computational
Biology, 5, 12, (December 2009), 1–18. doi: 10.1371/journal.pcbi.1000585. https:
//doi.org/10.1371/journal.pcbi.1000585.

