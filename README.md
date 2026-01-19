# SGMCI
Subgraph representation learning predicts multi-way chromatin interactions underlying cell type specificity determination

![image](https://github.com/user-attachments/assets/2e9ffd65-620e-40b5-af7a-3313028fcdf6)

# Requirements:
You'll need to install the following packages in order to run the codes.
- python 3.9.6
- pytorch 1.9.0
- torch-geometric 1.7.2
- scikit-learn 1.3.2
- networkx 3.2.1
- numpy 1.22.3
- pandas 1.3.5
- umap-learn 0.5.6

## Datasets

We provided test data in this repository.

If you need more, please contact us (xujinsheng@mail.kiz.ac.cn) to obtain all data used in the study.

## Usages
```
python SGMCI.py --use_struc --use_seed --repeat 10 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode MIX --test_chr 'chr1' --genome hg38 --binsize 1Mb
```
