import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from scipy.special import comb, perm
from itertools import combinations

# maxdist阈值的确定（3-way正样本）：
# 1. 计算三个点两两之间的空间距离的`平均值`作为三点间的平均距离
# 2. 计算不同的maxdist三点在`所有细胞中共定位的比例`与`MCI freq`的`相关系数`：100/150/200/250/300/350/400/450/500nm
# 3. 取`相关系数最大`时对应的maxdist作为最终阈值



def calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict, max_dist_threshold=250):

    haploids = list(snapfish_imputed_coor_dict[bin_1])
    dist_avg_list = []

    for haploid in haploids:
        coor_bin_1 = snapfish_imputed_coor_dict[bin_1][haploid]
        coor_bin_2 = snapfish_imputed_coor_dict[bin_2][haploid]
        coor_bin_3 = snapfish_imputed_coor_dict[bin_3][haploid]

        dist_12 = np.linalg.norm(coor_bin_1 - coor_bin_2)
        dist_13 = np.linalg.norm(coor_bin_1 - coor_bin_3)
        dist_23 = np.linalg.norm(coor_bin_2 - coor_bin_3)

        dist_avg = np.mean([dist_12, dist_13, dist_23])

        dist_avg_list.append(dist_avg)

    dist_avg_list = np.array(dist_avg_list)

    colocalization_ratio = dist_avg_list[dist_avg_list < max_dist_threshold].shape[0] / len(haploids)

    return colocalization_ratio


############
### HiPore-C mESC 5kb
node2bin = np.load('/data/xujs/Project/DeepLearning/MCIP/Results/preprocess_results/mm10.25kb.node2bin.npy', allow_pickle=True).item()
bin2node = np.load('/data/xujs/Project/DeepLearning/MCIP/Results/preprocess_results/mm10.25kb.bin2node.npy', allow_pickle=True).item()
mci_data_dir = '/data/xujs/Project/DeepLearning/MCIP/Results/HiPore-C_mESC/25kb'
data = np.load(f'{mci_data_dir}/all_3_NotDecompose_subgraph.npy')
data_freq = np.load(f'{mci_data_dir}/all_3_NotDecompose_subgraph_freq.npy')

####
FISH_data_dir = '/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2024-07-23_DNA_seqFISH+_validation/SnapFISH-IMPUTE'
### read coor file.
df_fish_snapfish_imputed_coor = pd.read_table(f'{FISH_data_dir}/output/recover_coor_mESCs_25kb_seqFISH.txt')
df_fish_linear_imputed_coor = pd.read_table(f'{FISH_data_dir}/output/linear_coor_mESCs_25kb_seqFISH.txt')
df_fish_raw_coor = pd.read_table(f'{FISH_data_dir}/data/mESC_seqFISH_25kb_coor_wnan.txt')

### read anno file.
df_fish_anno = pd.read_table(f'{FISH_data_dir}/data/mESC_seqFISH_25kb_ann.txt')
df_fish_anno['chrom'] = [ f'chr{i}' for i in df_fish_anno['region']]
df_fish_anno['chrom'] = df_fish_anno['chrom'].replace("chr20", "chrX")
df_fish_anno['bin'] = df_fish_anno['chrom'] + ':' + df_fish_anno['start'].map(str)



### begin calc ...

# df_chr1 = pd.DataFrame(columns=['bin1', 'bin2', 'bin3', 'freq', 'colocalization_ratio'])
snapfish_imputed_coor_dict = np.load('./SnapFISH-IMPUTE/output/mESCs_25kb_seqFISH_Snapfish_Imputed_Coor_dict.npy', allow_pickle=True).item()

out_dir = '/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2024-07-23_DNA_seqFISH+_validation/MCI_Freq_vs_Colocalization_ratio'


# max_dist_thresholds = [300]
# chroms = ['chr1']

# max_dist_thresholds = [100, 150, 200, 250, 300, 350, 400, 450, 500]
chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
            'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19']

for chrom in chroms:
    print(chrom)
    chrom_fish_bin_list = df_fish_anno.loc[df_fish_anno.chrom == chrom, 'bin'].tolist()

    df_chrom = pd.DataFrame(columns=['bin1', 'bin2', 'bin3', 'freq',
                                    'maxdist_50nm_colocalization_ratio',
                                    'maxdist_100nm_colocalization_ratio',
                                    'maxdist_150nm_colocalization_ratio',
                                    'maxdist_200nm_colocalization_ratio',
                                    'maxdist_250nm_colocalization_ratio',
                                    'maxdist_300nm_colocalization_ratio',
                                    'maxdist_350nm_colocalization_ratio',
                                    'maxdist_400nm_colocalization_ratio',
                                    'maxdist_450nm_colocalization_ratio',
                                    'maxdist_500nm_colocalization_ratio'])
    # for max_dist_threshold in max_dist_thresholds:

    for comb_3_bin in tqdm(list(combinations(chrom_fish_bin_list, 3))):
        bin_1, bin_2, bin_3 = comb_3_bin[0], comb_3_bin[1], comb_3_bin[2]
        index = np.where((data == (bin2node[bin_1], bin2node[bin_2], bin2node[bin_3])).all(axis=1))
        # freq = data_freq[index]
        freq = [data_freq[index][0] if data_freq[index].shape[0] else 0][0]
        maxdist_50_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=50)
        maxdist_100_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=100)
        maxdist_150_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=150)
        maxdist_200_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=200)
        maxdist_250_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=250)
        maxdist_300_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=300)
        maxdist_350_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=350)
        maxdist_400_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=400)
        maxdist_450_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=450)
        maxdist_500_coloc_ratio = calc_colocalization_ratio(bin_1, bin_2, bin_3, coor_dict=snapfish_imputed_coor_dict, max_dist_threshold=500)
        df_chrom.loc[len(df_chrom.index)] = [bin_1, bin_2, bin_3, 
                                            freq,
                                            maxdist_50_coloc_ratio,
                                            maxdist_100_coloc_ratio, 
                                            maxdist_150_coloc_ratio,
                                            maxdist_200_coloc_ratio,
                                            maxdist_250_coloc_ratio,
                                            maxdist_300_coloc_ratio,
                                            maxdist_350_coloc_ratio,
                                            maxdist_400_coloc_ratio,
                                            maxdist_450_coloc_ratio,
                                            maxdist_500_coloc_ratio]

    df_chrom.to_csv(f'{out_dir}/mESC_25Kb_MCI_Freq_vs_Colocalization_ratio_{chrom}.csv', header=True, sep='\t', index=False)