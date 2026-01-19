#!/bin/bash
###
 # @Author: xujs
 # @Date: 2024-03-07 16:09:28
 # @LastEditors: xujs
 # @LastEditTime: 2026-01-19 14:29:12
 # @FilePath: /SGMCI/run_SGMCI.sh
 # @Contact me: hzaujsxu@163.com
 # @Description:
###

hg38_chroms=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10
        chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
        chr21 chr22 chrX)

mm10_chroms=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10
        chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
        chr21 chr22 chrX)

# chroms=(chr2L chr2R chr3L chr3R chr4 chrX)

######################################################################################
##############################  1. HiPore-C K562  ####################################
######################################################################################
# 1.1 HiPore-C K562 1Mb
######  ND
python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_K562_1Mb --ns_mode RNS --decompose ND --genome hg38 --binsize 1Mb
python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_K562_1Mb --ns_mode BNS --decompose ND --genome hg38 --binsize 1Mb
python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_K562_1Mb --ns_mode SNS --decompose ND --genome hg38 --binsize 1Mb
python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_K562_1Mb --ns_mode MIX --decompose ND --genome hg38 --binsize 1Mb

# 1.2 HiPore-C K562 500kb
######  ND
python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_K562_500kb --ns_mode RNS --decompose ND --genome hg38 --binsize 500kb
python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_K562_500kb --ns_mode BNS --decompose ND --genome hg38 --binsize 500kb
python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_K562_500kb --ns_mode SNS --decompose ND --genome hg38 --binsize 500kb
python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_K562_500kb --ns_mode MIX --decompose ND --genome hg38 --binsize 500kb

####################################################################################
############################  2. HiPore-C GM12878 ##################################
####################################################################################
2.1 HiPore-C GM12878 1Mb
#######  ND
for chrom in ${chroms[*]};do
    echo "${chrom}"
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode RNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode BNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode SNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode MIX --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
done

# 2.2 HiPore-C GM12878 500kb
# #######  ND
for chrom in ${chroms[*]};do
    echo "${chrom}"
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_GM12878_500kb --ns_mode RNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 500kb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_GM12878_500kb --ns_mode BNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 500kb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_GM12878_500kb --ns_mode SNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 500kb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_GM12878_500kb --ns_mode MIX --decompose ND --test_chr ${chrom} --genome hg38 --binsize 500kb
done

######################################################################################
##############################   3. HiPore-C hESC   ##################################
######################################################################################
## 3.1 HiPore-C hESC 1Mb
for chrom in ${hg38_chroms[*]};do
    echo "${chrom}"
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_hESC_1Mb --ns_mode RNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_hESC_1Mb --ns_mode BNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_hESC_1Mb --ns_mode SNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    # python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_hESC_1Mb --ns_mode MIX --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
done

######################################################################################
##############################   4. HiPore-C mESC   ##################################
######################################################################################
## 4.1 HiPore-C mESC 1Mb
for chrom in ${mm10_chroms[*]};do
    echo "${chrom}"
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_mESC_1Mb --ns_mode RNS --decompose ND --test_chr ${chrom} --genome mm10 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_mESC_1Mb --ns_mode BNS --decompose ND --test_chr ${chrom} --genome mm10 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 1 --dataset HiPore-C_mESC_1Mb --ns_mode SNS --decompose ND --test_chr ${chrom} --genome mm10 --binsize 1Mb
    # python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset HiPore-C_mESC_1Mb --ns_mode MIX --decompose ND --test_chr ${chrom} --genome mm10 --binsize 1Mb
done

# #####################################################################################
# ##############################   5. RNAPII ChIA-Drop S2  ############################
# #####################################################################################
# ## 5.1 RNAPII ChIA-Drop dm3 25kb
# for chrom in ${chroms[*]};do
#     echo "${chrom}"
#     python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset RNAPII-ChIA-Drop_S2_25kb --ns_mode MIX --decompose ND --test_chr ${chrom} --genome dm3 --binsize 25kb
# done

#####################################################################################
#################################  6. ChIA-Drop S2  #################################
#####################################################################################
# ## 6.1 ChIA-Drop dm3 25kb
# for chrom in ${chroms[*]};do
#     echo "${chrom}"
#     python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 1 --device 0 --dataset ChIA-Drop_S2_25kb --ns_mode MIX --decompose ND --test_chr ${chrom} --genome dm3 --binsize 25kb
# done