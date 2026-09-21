setwd('/data/xujs/Project/HiC2PoreC/code/H2P/results')

library(openxlsx)
library(ggplot2)
library(reshape2)

### 读取所有数据的H2P性能
## GM12878 K562 hESC mESC
## 1Mb 100kb 5kb
## RNS BNS SNS MIX
df_H2P_hg38_RNS_MP <- read.table('H2P_HiPore-C_hg38.Chrom_RNS_Performance.txt', header = TRUE, sep='\t')[1:23, ]
df_H2P_hg38_BNS_MP <- read.table('H2P_HiPore-C_hg38.Chrom_BNS_Performance.txt', header = TRUE, sep='\t')[1:23, ]
df_H2P_hg38_SNS_MP <- read.table('H2P_HiPore-C_hg38.Chrom_SNS_Performance.txt', header = TRUE, sep='\t')[1:23, ]
df_H2P_hg38_MIX_MP <- read.table('H2P_HiPore-C_hg38.Chrom_MIX_Performance.txt', header = TRUE, sep='\t')[1:23, ]
df_H2P_mm10_RNS_MP <- read.table('H2P_HiPore-C_mm10.Chrom_RNS_Performance.txt', header = TRUE, sep='\t')[1:20, ]
df_H2P_mm10_BNS_MP <- read.table('H2P_HiPore-C_mm10.Chrom_BNS_Performance.txt', header = TRUE, sep='\t')[1:20, ]
df_H2P_mm10_SNS_MP <- read.table('H2P_HiPore-C_mm10.Chrom_SNS_Performance.txt', header = TRUE, sep='\t')[1:20, ]
df_H2P_mm10_MIX_MP <- read.table('H2P_HiPore-C_mm10.Chrom_MIX_Performance.txt', header = TRUE, sep='\t')[1:20, ]


### GM12878
df_GM12878_MP <- data.frame(chrom=df_H2P_hg38_RNS_MP$Chrom,
                             RNS_1Mb = df_H2P_hg38_RNS_MP$HiPore.C_GM12878_1Mb,
                             RNS_100Kb = df_H2P_hg38_RNS_MP$HiPore.C_GM12878_100kb,
                             RNS_5Kb = df_H2P_hg38_RNS_MP$HiPore.C_GM12878_5kb,
                             BNS_1Mb = df_H2P_hg38_BNS_MP$HiPore.C_GM12878_1Mb,
                             BNS_100Kb = df_H2P_hg38_BNS_MP$HiPore.C_GM12878_100kb,
                             BNS_5Kb = df_H2P_hg38_BNS_MP$HiPore.C_GM12878_5kb,
                             SNS_1Mb = df_H2P_hg38_SNS_MP$HiPore.C_GM12878_1Mb,
                             SNS_100Kb = df_H2P_hg38_SNS_MP$HiPore.C_GM12878_100kb,
                             SNS_5Kb = df_H2P_hg38_SNS_MP$HiPore.C_GM12878_5kb,
                             MIX_1Mb = df_H2P_hg38_MIX_MP$HiPore.C_GM12878_1Mb,
                             MIX_100Kb = df_H2P_hg38_MIX_MP$HiPore.C_GM12878_100kb,
                             MIX_5Kb = df_H2P_hg38_MIX_MP$HiPore.C_GM12878_5kb)
df_GM12878_MP <- melt(df_GM12878_MP, id.vars = 'chrom', variable.name = 'type', value.name = 'AUC')
df_GM12878_MP$type <- as.character(df_GM12878_MP$type)
df_GM12878_MP$NS <- unlist(lapply(df_GM12878_MP$type, function(x) {strsplit(x, '_')[[1]][1]}))
df_GM12878_MP$Res <- unlist(lapply(df_GM12878_MP$type, function(x) {strsplit(x, '_')[[1]][2]}))

df_GM12878_MP$AUC = as.numeric(df_GM12878_MP$AUC)
df_GM12878_MP$Res = factor(df_GM12878_MP$Res, levels=c('1Mb', '100Kb', '5Kb'))
df_GM12878_MP$NS = factor(df_GM12878_MP$NS, levels = c('RNS', 'BNS', 'SNS', 'MIX'))

pdf('GM12878_Res_NS_Chrom_AUC.boxplot_scatter.pdf', height=4, width = 6)
ggplot(df_GM12878_MP, aes(x=Res, y=AUC, color=NS))+
  stat_boxplot(geom = "errorbar", width=0.4, position = position_dodge(0.8))+
  geom_boxplot(width=0.7, position = position_dodge(0.8), outlier.colour = NA)+
  # geom_violin(wdith=0.5, position = position_dodge(0.5))+
  geom_jitter(aes(fill=NS), shape=21, size=2, alpha=0.8, show.legend = TRUE, position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8))+
  scale_fill_manual(values = c('#006934', '#0b318f', '#ea5514', '#a44393'))+
  scale_color_manual(values = c('#006934', '#0b318f', '#ea5514', '#a44393'))+
  scale_y_continuous(limits = c(0.6, 1), expand = c(0,0))+
  labs(x='', y='AUC score', title='GM12878')+
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=15),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.y = element_text(size=14, color="black"),
        axis.text.x = element_text(size=12, color="black"),
        axis.text.y = element_text(size=12, color="black"))
dev.off()
