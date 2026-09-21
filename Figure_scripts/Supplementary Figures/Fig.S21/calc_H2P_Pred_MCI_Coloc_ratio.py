import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm, trange
from scipy.special import comb, perm
from itertools import combinations

from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from scipy.stats import pearsonr, spearmanr 
from genomic_regions import as_region

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def convert_chrom_region(region):
    chrom, start, end = re.split(':|-|_', region)
    return f'{chrom}:{start}-{end}'

def complete_region(bin_chrom_start, bin_size=5000):
    '''
    bin_chrom_start: chr1:120000000
    output: chr1:120000000-120005000
    '''
    chrom, start = bin_chrom_start.split(':')
    start = int(start)
    end = start + bin_size
    region = f'{chrom}:{start}-{end}'
    region = as_region(region)

    return region

def mini_resolution_from_highRes_to_lowRes(region, highRes=5000, lowRes=25000):
    '''
    将高分辨率数据降至低分辨率: e.g. 5kb -> 25kb
    '''
    chrom = region.chromosome
    start = int((region.start + highRes) / lowRes) * lowRes
    end = start + lowRes

    return f'{chrom}:{start}'

def generate_same_num_pos(selected_mcis, o3_mci):
    idx = np.random.choice(o3_mci.shape[0], size=selected_mcis.shape[0])
    random_o3_mci = o3_mci[idx]

    return random_o3_mci


def calc_colocalization_ratio(bin_1, bin_2, bin_3, snapfish_imputed_coor_dict, max_dist_threshold=250):

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


def get_data_colco_ratio_list(data_pos, max_dist_threshold=150):
    coloc_ratio_list = []
    for mci in tqdm(data_pos):
        region_mci = [node2bin_5kb[node] for node in mci]
        region_25k_mci = [mini_resolution_from_highRes_to_lowRes(complete_region(region_)) for region_ in region_mci]
        
        try:
            coloc_ratio = calc_colocalization_ratio(region_25k_mci[0], region_25k_mci[1], region_25k_mci[2], snapfish_imputed_coor_dict, max_dist_threshold=max_dist_threshold)
        except KeyError:
            continue

        coloc_ratio_list.append(coloc_ratio)
    
    return coloc_ratio_list


#### run code.
pred_data_dir = '/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2026-08-24_Revised_Results/2026-09-03_predict_mESC_New_O4_MCI_5kb/chrom_region_Unobserved_MCI'
# snapfish_imputed_coor_dict = np.load('/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2024-07-23_DNA_seqFISH+_validation/SnapFISH-IMPUTE/output/mESCs_25kb_seqFISH_Snapfish_Imputed_Coor_dict.npy', allow_pickle=True).item()
snapfish_imputed_coor_dict = np.load('/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2024-07-23_DNA_seqFISH+_validation/SnapFISH-IMPUTE/output/mESCs_25kb_seqFISH_raw_Coor_dict.npy', allow_pickle=True).item()

node2bin = np.load('/data/xujs/Project/DeepLearning/MCIP/Results/preprocess_results/mm10.25kb.node2bin.npy', allow_pickle=True).item()
bin2node = np.load('/data/xujs/Project/DeepLearning/MCIP/Results/preprocess_results/mm10.25kb.bin2node.npy', allow_pickle=True).item()

node2bin_5kb = np.load('/data/xujs/Project/DeepLearning/MCIP/Results/preprocess_results/mm10.5kb.node2bin.npy', allow_pickle=True).item()
bin2node_5kb = np.load('/data/xujs/Project/DeepLearning/MCIP/Results/preprocess_results/mm10.5kb.bin2node.npy', allow_pickle=True).item()

bin_size = 5000
max_dist_threshold=150

# fish_20_regions =  ['chr1:135600000-137100000', 'chr2:109000000-110575000', 'chr3:7675000-9325000',    'chr4:89300000-91025000',   'chr5:131400000-132900000',
#                     'chr6:49375000-50925000',   'chr7:44600000-46100000',   'chr8:42700000-44425000',  'chr9:119300000-120800000', 'chr10:75400000-76900000',
#                     'chr11:97425000-98925000',  'chr12:61900000-64150000',  'chr13:21475000-23800000', 'chr14:65400000-66900000',  'chr15:61000000-62525000',
#                     'chr16:90500000-92000000',  'chr17:62425000-63925000',  'chr18:71200000-72700000', 'chr19:12550000-14900000'] # 'chr20:75325457-77025457

# chrom_regions =    ['chr1_120000000-140000000', 'chr2_100000000-120000000', 'chr3_0-20000000',         'chr4_80000000-100000000',  'chr5_120000000-140000000',
#                     'chr6_40000000-60000000',   'chr7_40000000-60000000',   'chr8_40000000-60000000',  'chr9_100000000-120000000', 'chr10_60000000-80000000',
#                     'chr11_80000000-100000000', 'chr12_60000000-80000000',  'chr13_20000000-40000000', 'chr14_60000000-80000000',  'chr15_60000000-80000000',
#                     'chr16_80000000-98207768',  'chr17_60000000-80000000',  'chr18_60000000-80000000', 'chr19_0-20000000'] # 'chrX_60000000-80000000'

fish_20_regions =  ['chr1:135600000-137100000']
chrom_regions = ['chr1_120000000-140000000']

# fish_20_regions =  ['chr15_60000000-80000000']
# chrom_regions = ['chr1_120000000-140000000']

## data节点id需要为全基因组内的总id
mm10_5kb_chrom_region_node_num = pd.read_table('/data/xujs/Project/HiC2PoreC/code/H2P/dataset/mm10.5kb.chrom_region.node_num.txt')

for i in range(len(fish_20_regions)):
    fish_region = fish_20_regions[i]
    chrom_region = chrom_regions[i]
    print(f'{chrom_region} ...')

    fish_region_ = as_region(fish_region)
    chrom_region_ = as_region(convert_chrom_region(chrom_region))
    
    # out_file = f'/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2024-07-23_DNA_seqFISH+_validation/H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_{max_dist_threshold}nm_{chrom_region}.txt'
    out_file = f'/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2026-08-24_Revised_Results/2026-08-30_order4_raw_DNAseqFISH+_validation/H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_{max_dist_threshold}nm_{chrom_region}.txt'
    # if os.path.exists(out_file):
    #     continue

    idx = mm10_5kb_chrom_region_node_num.loc[mm10_5kb_chrom_region_node_num['chrom_region'] == chrom_region].index[0]
    node_id_start = sum(mm10_5kb_chrom_region_node_num.iloc[:idx, 1].to_list())
    ###################       Step1: 加载模型预测数据.        ################### 
    #### Note: *2.npy文件是第二次跑的, 同时保存了MCI数据
    data = np.load(f'{pred_data_dir}/HiPore-C_mESC_5kb_{chrom_region}_Unobserved_4way_MCI.npy')
    data = data + node_id_start - 1
    pred_score = np.load(f'{pred_data_dir}/HiPore-C_mESC_5kb_{chrom_region}_Unobserved_4way_MCI_pred_score.npy').reshape(-1)

    ###################  Step2: 划分：H2P预测是否为3way-MCI.  ################### 
    data_pos = data[(pred_score > 0.9) & (pred_score <= 0.999)]
    data_neg = data[(pred_score < 0.1) & (pred_score >= 0.001)]

    # node_start =int( chrom_region_.start / bin_size)+1
    # node_end =int( chrom_region_.end / bin_size)

    # data_pos = data_pos + node_start - 1
    # data_neg = data_neg + node_start - 1 

    s, e = bin2node_5kb[f'{fish_region_.chromosome}:{fish_region_.start}'], bin2node_5kb[f'{fish_region_.chromosome}:{fish_region_.end}']

    data_pos = data_pos[np.all((data_pos >= s) & (data_pos < e), axis=1)]
    data_neg = data_neg[np.all((data_neg >= s) & (data_neg < e), axis=1)]


    ###################  Step3: 计算三组样本的coloc ratio.  ################### 
    try:
        ### Group1: pos coloc ratio.
        pos_coloc_ratio_list = get_data_colco_ratio_list(data_pos, max_dist_threshold=max_dist_threshold)
        ### Group2: neg coloc ratio.
        data_neg = generate_same_num_pos(data_pos, data_neg)
        neg_coloc_ratio_list = get_data_colco_ratio_list(data_neg, max_dist_threshold=max_dist_threshold)
        ### Group3: random coloc ratio.
        fish_bin_list = np.arange(np.min(data_pos), np.max(data_pos))
        data_random = np.array(list(combinations(fish_bin_list, 3)))
        data_random = generate_same_num_pos(data_pos, data_random)
        random_coloc_ratio_list = get_data_colco_ratio_list(data_random, max_dist_threshold=max_dist_threshold)

        ## save coloc ratio file.
        df = pd.DataFrame({
                'Group': ['Pos']*len(pos_coloc_ratio_list) + ['Neg']*len(neg_coloc_ratio_list) + ['Random']*len(random_coloc_ratio_list),
                'Coloc_Ratio': pos_coloc_ratio_list + neg_coloc_ratio_list + random_coloc_ratio_list
            })
        df.to_csv(out_file, header=True, index=False, sep='\t')

    except:
        continue