setwd('F:/HiC2PoreC/Analysis_Results/2024-05-23_Predict_New_MCI')

library(ggplot2)

### 1. Hi-C edge support.
# df_prob_group <- read.table('HiPore-C_GM12878_1Mb.Unobserved_MCI_O3_Hi-C_edge_prob_group.txt', header = TRUE)
df_prob_group <- read.table('HiPore-C_GM12878_1Mb.SGMCI_Unobserved_MCI_O3_Hi-C_edge_prob_group.txt', header = TRUE)
df_prob_group$edge_num <- as.character(df_prob_group$edge_num)
df_prob_group$edge_num <- factor(df_prob_group$edge_num, levels = rev(c("0", "1", "2", "3")))


pdf('./HiPore-C_GM12878_1Mb.Unobserved_MCI_O3_Hi-C_edge_prob_group.pdf', width=4.5, height=3.5)
ggplot(df_prob_group, aes(x=prob_group, weight=perc, fill=edge_num, group=edge_num))+
  geom_bar(position = "stack", color='black')+ ## stack -> dodge 分组柱状图
  # scale_fill_manual(values = rev(c('#adb8dd', '#d3b098', '#c68368', '#a34d3c')))+
  scale_fill_manual(values = rev(c('#989cc8', '#78a9cd', '#f7ab60', '#c93835')))+
  scale_y_continuous(expand = c(0,0))+
  labs(x='Prob group', y='Percentage', title='')+
  # theme_bw()+
  theme_classic()+
  theme(plot.title = element_text(size=14, colour='black', hjust = 0.5),
        axis.title = element_text(size = 13, colour = "black"),
        axis.text.x = element_text(size=10, color="black", angle = 30, vjust = 0.5),
        axis.text.y = element_text(size=10, color="black"))
dev.off()