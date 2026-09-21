#!/bin/bash
###
 # @Author: xujs
 # @Date: 2024-03-07 16:09:28
 # @LastEditors: xujs
 # @LastEditTime: 2026-09-21 14:29:12
 # @FilePath: /SGMCI/run_SGMCI.sh
 # @Contact me: xujinsheng@mail.kiz.ac.cn
 # @Description:
###

hg38_chroms=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10
        chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20
        chr21 chr22 chrX)

##########################################################################################
############################  e.g. HiPore-C GM12878 1Mb ##################################
##########################################################################################
####### HiPore-C GM12878 1Mb

chroms=('chr1')  ### take chr1 as example.

for chrom in ${chroms[*]};do
    echo "${chrom}"
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 10 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode RNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 10 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode BNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 10 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode SNS --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb
    python SGMCI.py --use_nodeid --use_seed --use_maxzeroone --repeat 10 --device 0 --dataset HiPore-C_GM12878_1Mb --ns_mode MIX --decompose ND --test_chr ${chrom} --genome hg38 --binsize 1Mb

done
