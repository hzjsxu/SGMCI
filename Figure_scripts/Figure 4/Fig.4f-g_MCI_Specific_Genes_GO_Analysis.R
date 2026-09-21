setwd('F:/HiC2PoreC/Analysis_Results/2024-06-20_cellType_specific_MCI')

library(biomaRt)
library(clusterProfiler)
library(openxlsx)
library(org.Hs.eg.db)
library(ggplot2)
library(gridExtra)
library(ggpubr)
library(DOSE)

GM12878_MCI_specific_genes <- read.table('GM12878_MCI_specific_genes.txt')$V1
K562_MCI_specific_genes <- read.table('K562_MCI_specific_genes.txt')$V1

ego_GM12878_MCI <- enrichGO(gene = GM12878_MCI_specific_genes, OrgDb = org.Hs.eg.db, keyType = 'SYMBOL', ont = 'ALL', pAdjustMethod='BH', pvalueCutoff=0.05, qvalueCutoff=0.1)
ego_K562_MCI <- enrichGO(gene = K562_MCI_specific_genes, OrgDb = org.Hs.eg.db, keyType = 'SYMBOL', ont = 'ALL', pAdjustMethod='BH', pvalueCutoff=0.05, qvalueCutoff=0.1)

ego_GM12878_go <- as.data.frame(ego_GM12878_MCI)
ego_GM12878_go$FoldEnrichment <- parse_ratio(ego_GM12878_go$GeneRatio)/parse_ratio(ego_GM12878_go$BgRatio)
ego_K562_go <- as.data.frame(ego_K562_MCI)
ego_K562_go$FoldEnrichment <- parse_ratio(ego_K562_go$GeneRatio)/parse_ratio(ego_K562_go$BgRatio)

write.csv(ego_GM12878_go, 'GM12878_MCI_specific_genes.GO_result.csv', row.names = F, quote = F)
write.csv(ego_K562_go, 'K562_MCI_specific_genes.GO_result.csv', row.names = F, quote = F)

ego_GM12878_go_bp <- ego_GM12878_go[(ego_GM12878_go$ONTOLOGY == 'BP') & (ego_GM12878_go$qvalue < 0.001) & (ego_GM12878_go$FoldEnrichment > 2),]
ego_K562_go_bp <- ego_K562_go[(ego_K562_go$ONTOLOGY == 'BP') & (ego_K562_go$qvalue < 0.01) & (ego_K562_go$FoldEnrichment > 2),]
ego_GM12878_go_cc <- ego_GM12878_go[(ego_GM12878_go$ONTOLOGY == 'CC'), ]
ego_K562_go_cc <- ego_K562_go[(ego_K562_go$ONTOLOGY == 'CC') & (ego_K562_go$qvalue < 0.01), ]
ego_GM12878_go_mf <- ego_GM12878_go[(ego_GM12878_go$ONTOLOGY == 'MF'), ]
ego_K562_go_mf <- ego_K562_go[(ego_K562_go$ONTOLOGY == 'MF'), ]


plot_GO_bar <- function(df_go, fill_color='grey', xlabel='BP'){
  df_go$ID <- factor(df_go$ID, levels = rev(df_go$ID))
  ggplot(data=df_go, aes(x=ID, y=-log10(pvalue)))+
    geom_bar(stat = 'identity', width=0.7, fill=fill_color)+
    geom_text(aes(y=0, label=Description, hjust='left'),size=5)+
    # geom_point(aes(size=Count, color=-log10(qvalue)))+
    coord_flip()+
    scale_y_continuous(expand = c(0,0))+
    labs(x=xlabel, y='-log10 (p-value)', title = '')+
    # theme_bw()+
    theme_classic()+
    theme(plot.title = element_text(size=14, colour='black', hjust = 0.5),
          axis.title = element_text(size = 13, colour = "black"),
          axis.text.y = element_blank(),
          axis.text = element_text(size=12, color="black"),
          axis.line.y = element_blank(),
          axis.ticks.y = element_blank())
  
}

p_gm12878_go_BP <- plot_GO_bar(ego_GM12878_go_bp, fill_color = '#e1b6a3', xlabel = 'Biological process')
p_gm12878_go_CC <- plot_GO_bar(ego_GM12878_go_cc, fill_color = '#c2df9f', xlabel = 'Cellular component')
p_gm12878_go_MF <- plot_GO_bar(ego_GM12878_go_mf, fill_color = '#f0f1a6', xlabel = 'Molecular function')
p_k562_go_BP <- plot_GO_bar(ego_K562_go_bp, fill_color = '#e1b6a3', xlabel = 'Biological process')
p_k562_go_CC <- plot_GO_bar(ego_K562_go_cc, fill_color = '#c2df9f', xlabel = 'Cellular component')
p_k562_go_MF <- plot_GO_bar(ego_K562_go_mf, fill_color = '#f0f1a6', xlabel = 'Molecular function')

pdf('MCI_Specific_Genes_GO_Barplot.pdf', height = 12, width = 12)
ggarrange(p_gm12878_go_BP, p_k562_go_BP,
          p_gm12878_go_CC, p_k562_go_CC, 
          p_gm12878_go_MF, p_k562_go_MF,
          ncol=2, nrow=3)
dev.off()
