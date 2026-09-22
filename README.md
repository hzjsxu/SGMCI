# SGMCI
Subgraph representation learning predicts multi-way chromatin interactions underlying cell type specificity determination

![image](https://github.com/user-attachments/assets/2e9ffd65-620e-40b5-af7a-3313028fcdf6)

Multi-way chromatin interactions (MCIs) orchestrate higher-order genome organization, yet their genome-wide inference from Hi-C remains challenging. We present SGMCI, a subgraph representation learning framework that casts MCI prediction as subgraph classification on Hi-C graphs. SGMCI integrates graph autoencoder-derived node embeddings, graph convolutional message passing, mean pooling, and MLP decoding. Chromosome-level leave-one-out cross-validation prevents transductive leakage. Across four human and mouse cell lines and three resolutions, SGMCI achieves superior performance. Predictions are validated by DNA seqFISH+. Separately, cell-type-specific MCIs identified by SGMCI are enriched for cell-identity genes. SGMCI offers a scalable route to decode 3D genome architecture from Hi-C data.

## Requirements:
You'll need to install the following packages in order to run the codes.
- python 3.9.6
- pytorch 1.9.0
- torch-geometric 1.7.2
- scikit-learn 1.3.2
- networkx 3.2.1
- numpy 1.22.3
- pandas 1.3.5
- umap-learn 0.5.6

We recommend using conda (the project root provides environment.yml):
```
# In the project root
conda env create -f environment.yml
conda activate SGMCI
```

## Data preparation
By default, scripts read data from the following directory: `dataset/HiPore-C_<name>_<resolution>`, where `<name>` could be GM12878, K562, hESC, and mESC; `<resolution>` could be 1Mb, 100kb, and 5kb. In this repository, we use HiPore-C_GM12878_1Mb as default.
The input data includes:
- `ND_MIX_subgraphs.pth`: consists of 4 columns: (1) MCI (Multi-way Chromatin Interaction): Composed of node IDs connected by hyphens. (2) Label: Binary label indicating positive [1] or negative [0] samples. (3) Data type: Specifies whether the sample belongs to the training, testing, or validation set (train, test, or val). (4) (2)Weight: The weight value associated with the MCI.
- `edge_list.txt`: the first two columns represent the IDs of a pair of nodes, and the third column indicates the interaction strength between them.
- `hg38.1Mb.node_num.txt`：the first column specifies the chromosome name, and the second column indicates the number of nodes associated with that chromosome. 


## Datasets

We provided test data in this repository.

If you need more, please contact us (xujinsheng@mail.kiz.ac.cn) to obtain all data used in the study.

## Usage
You can directly run `run_SGMCI.sh` to get results of 4 negative sampling strategies, or run the code below to test the code:
```
python SGMCI.py --use_struc --use_seed --repeat 10 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode MIX --test_chr 'chr1' --genome hg38 --binsize 1Mb
```

## Output
Output are saved to `results/<dataset>` directory by default. It includes:
- `<dataset>.tsv`: SGMCI's prediction score for each ccandidate subgraph (MCI).
- `<dataset>.pt`: the best model saved.


## Reproducing main figures

The directory Figure_scripts contains the source data and plotting scripts used in the SGMCI manuscript.

### Contact

- Issues: please submit via repository issues (include error logs, commands, environment info, and reproduction)
