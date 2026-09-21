setwd('F:/HiC2PoreC/Analysis_Results/2024-05-23_Predict_New_MCI')

library(ggplot2)

### 1. scHi-C edge support.
# df_schic_prob_group <- read.table('HiPore-C_GM12878_1Mb.Unobserved_MCI_O3_scHi-C_edge_prob_group.txt', header = TRUE)
df_schic_prob_group <- read.table('HiPore-C_GM12878_1Mb.SGMCI_Unobserved_MCI_O3_scHi-C_edge_prob_group.txt', header = TRUE)
df_schic_prob_group$schic_support <- factor(df_schic_prob_group$schic_support, levels = c("1", "0"))

pdf('./HiPore-C_GM12878_1Mb.Unobserved_MCI_O3_scHi-C_prob_group.pdf', width=4.5, height=3.5)
ggplot(df_schic_prob_group[df_schic_prob_group$schic_support == 1,],
       aes(x=prob_group, weight=perc, fill=schic_support))+
  geom_bar(position = "dodge", color='black')+ ## stack -> dodge 分组柱状图
  # scale_fill_manual(values = rev(c('#adb8dd', '#d3b098', '#c68368', '#a34d3c')))+
  scale_fill_manual(values = c('#f7ab60', '#989cc8', '#78a9cd', '#c93835'))+
  scale_y_continuous(expand = c(0,0), limits = c(0, 0.25))+
  labs(x='Prob group', y='Percentage', title='')+
  # theme_bw()+
  theme_classic()+
  theme(plot.title = element_text(size=14, colour='black', hjust = 0.5),
        axis.title = element_text(size = 13, colour = "black"),
        axis.text.x = element_text(size=10, color="black", angle = 30, vjust = 0.5),
        axis.text.y = element_text(size=10, color="black"))
dev.off()
