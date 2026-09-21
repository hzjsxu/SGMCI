setwd('/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2024-06-07_Node_feature/')

library(ggplot2)
library(reshape2)
library(ggpubr)

### 1. AUC
df_model_variants_AUC <- data.frame(cellType=c('GM12878', 'K562', 'hESC', 'mESC'),
                                    H2P =  c(0.9423, 0.9417, 0.9177, 0.8691),
                                    Epi_H2P=c(0.9473, 0.9433, 0.9204, 0.8784),
                                    Seq_H2P=c(0.9501, 0.9447, 0.9200, 0.8998),
                                    ES_H2P=c(0.9512, 0.9509, 0.9214, 0.9017))
df_model_variants_AUC <- melt(df_model_variants_AUC, id.vars = 'cellType', variable.name = 'Model_variants', value.name = 'AUC')
df_model_variants_AUC$cellType <- factor(df_model_variants_AUC$cellType, levels = c('GM12878', 'K562', 'hESC', 'mESC'))

p_AUC <- ggplot(data = df_model_variants_AUC, aes(x=cellType, y=AUC-0.8, fill=Model_variants))+
  geom_bar(stat = 'identity', position = position_dodge(0.7), width=0.5, color='black')+
  scale_fill_manual(values = c('#38749f', '#a2c5d3', '#efb094', '#b43838'))+
  # geom_text(aes(y=0, label=Description, hjust='left'),size=5)+
  # geom_point(aes(size=Count, color=-log10(qvalue)))+
  # coord_flip()+
  scale_y_continuous(expand = c(0,0), limits = c(0, 0.2), breaks = seq(0, 0.2, 0.05), labels = seq(0, 0.2, 0.05) + 0.8)+
  labs(x='', y='AUC', title = '')+
  # theme_bw()+
  theme_classic()+
  theme(plot.title = element_text(size=14, colour='black', hjust = 0.5),
        axis.title = element_text(size = 13, colour = "black"),
        # axis.text.y = element_blank(),
        axis.text = element_text(size=12, color="black"),
        # axis.line.y = element_blank(),
        # axis.ticks.y = element_blank()
        )
  

### 2. ACC
df_model_variants_ACC <- data.frame(cellType=c('GM12878', 'K562', 'hESC', 'mESC'),
                                    H2P =  c(0.8889, 0.8757, 0.8329, 0.8091),
                                    Epi_H2P=c(0.8912, 0.8808, 0.8418, 0.8151),
                                    Seq_H2P=c(0.8906, 0.8845, 0.8411, 0.8219),
                                    ES_H2P=c(0.8915, 0.8862, 0.8470, 0.8273))
df_model_variants_ACC <- melt(df_model_variants_ACC, id.vars = 'cellType', variable.name = 'Model_variants', value.name = 'ACC')
df_model_variants_ACC$cellType <- factor(df_model_variants_ACC$cellType, levels = c('GM12878', 'K562', 'hESC', 'mESC'))

p_ACC <- ggplot(data = df_model_variants_ACC, aes(x=cellType, y=ACC-0.7, fill=Model_variants))+
  geom_bar(stat = 'identity', position = position_dodge(0.7), width=0.5, color='black')+
  scale_fill_manual(values = c('#38749f', '#a2c5d3', '#efb094', '#b43838'))+
  # geom_text(aes(y=0, label=Description, hjust='left'),size=5)+
  # geom_point(aes(size=Count, color=-log10(qvalue)))+
  # coord_flip()+
  scale_y_continuous(expand = c(0,0), limits = c(0, 0.3), breaks = seq(0, 0.3, 0.1), labels = seq(0, 0.3, 0.1) + 0.7)+
  labs(x='', y='ACC', title = '')+
  # theme_bw()+
  theme_classic()+
  theme(plot.title = element_text(size=14, colour='black', hjust = 0.5),
        axis.title = element_text(size = 13, colour = "black"),
        # axis.text.y = element_blank(),
        axis.text = element_text(size=12, color="black"),
        # axis.line.y = element_blank(),
        # axis.ticks.y = element_blank()
  )

pdf('Model_Variants_AUC_ACC.BarPlot.pdf', height = 6, width = 9)
ggarrange(p_AUC, p_ACC, nrow=2)
dev.off()
