library(ggplot2)
library(dplyr)

setwd('/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2024-07-23_DNA_seqFISH+_validation')

df_coloc_ratio_freq <- read.table('Correlation_between_Coloc_ratio_and_Freq.txt', header = T)

df_coloc_ratio_freq$max_dist <- sapply(df_coloc_ratio_freq$max_dist_coloc_ratio, function(x) strsplit(x, split = '_')[[1]][2])
df_coloc_ratio_freq$max_dist <- factor(df_coloc_ratio_freq$max_dist, levels = unique(df_coloc_ratio_freq$max_dist))

chroms <- c('chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
            'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19')


## 1. 染色体水平的相关性
plot_pearson_cor_point_line <- function(df_coloc_ratio_freq, chrom='chr1'){
    ggplot(data = df_coloc_ratio_freq[df_coloc_ratio_freq$chrom == chrom,],
            aes(x=1:10, y=pearson_R))+
    geom_line(linewidth=0.8, color='#010180')+
    geom_point(size=2, color='#010180')+
    scale_x_continuous(breaks = seq(1, 10), labels = seq(50, 500, 50))+
    scale_y_continuous(limits = c(0, 0.8), breaks = seq(0, 0.8, 0.2))+
    labs(title=chrom, x='Threshold of colocalization (nm)', y='Pearson_R')+
    theme_bw()+
    theme(
      strip.background = element_rect(color=NA, fill='white', linewidth=1, linetype = 'solid'),
      strip.text = element_text(size=12, color='black'),
      panel.spacing = unit(1, 'cm'),
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 14, vjust = 0),
      axis.text = element_text(size=12, color='black'),
      axis.text.x = element_text(size=12, color='black', angle = 0),
      axis.title = element_text(size=14),
      legend.position = 'none')
}


pdf('Pearson_Correlation_between_Coloc_ratio_and_Freq.chrom_point_line.pdf',width = 4*5, height = 4*3)
grid.newpage()
pushViewport(viewport(layout = grid.layout(4,5)))
a = 0
for (chrom in chroms){
  cat(chrom, sep = '\n')
  p_chrom <- plot_pearson_cor_point_line(df_coloc_ratio_freq, chrom=chrom)
  a = a+1
  row = ifelse(a%%5 == 0, (a-1)%/%5+1, a%/%5+1)
  col = ifelse(a%%5 == 0, 5, a%%5)
  print(p_chrom, vp=viewport(layout.pos.row = row, layout.pos.col = col))
}
dev.off()


## 2. 全基因组水平的箱线图：不同max_dist下各染色体相关性
pdf('Pearson_Correlation_between_Coloc_ratio_and_Freq.WholeGenome_boxplot.pdf',width=8, height=5)
ggplot(data = df_coloc_ratio_freq, aes(max_dist, y=pearson_R, fill=max_dist))+
  geom_boxplot(width=0.5, fill='white', color='#5b7aa2', linewidth=1.5)+
  geom_jitter(width = 0.2, color='#93a8c2', size=2)+
  labs(title='mESC chr1~chr19', x='Threshold of colocalization (nm)', y='Correlation coefficient')+
  theme_bw()+
  theme(
    strip.background = element_rect(color=NA, fill='white', linewidth=1, linetype = 'solid'),
    strip.text = element_text(size=12, color='black'),
    panel.spacing = unit(1, 'cm'),
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 14, vjust = 0),
    axis.text = element_text(size=12, color='black'),
    axis.text.x = element_text(size=12, color='black', angle = 0),
    axis.title = element_text(size=14),
    legend.position = 'none')
dev.off()


## 3. 染色体上H2P预测为正/负的coloc_ratio.

## 以chr1为例：

chrom_regions <- c('chr1_120000000-140000000',
                   'chr2_100000000-120000000',
                   'chr3_0-20000000',
                   'chr4_80000000-100000000',
                   'chr5_120000000-140000000',
                   'chr6_40000000-60000000',
                   'chr8_40000000-60000000',
                   'chr9_100000000-120000000',
                   'chr10_60000000-80000000',
                   'chr11_80000000-100000000',
                   'chr12_60000000-80000000',
                   'chr13_20000000-40000000',
                   'chr14_60000000-80000000',
                   'chr15_60000000-80000000',
                   'chr16_80000000-98207768',
                   'chr17_60000000-80000000',
                   'chr18_60000000-80000000',
                   'chr19_0-20000000')


#############################################################################

plot_H2P_coloc_ratio_boxplot <- function(df, chrom_region, y_lim=c(0, 0.035), x_labels=c('Prob>0.5', 'Random', 'Prob<0.5')){
  chrom <- strsplit(chrom_region, '_')[[1]][1]
  ggplot(data = df, aes(x=Group, y=Coloc_Ratio, color=Group))+
    geom_boxplot(width=0.5, fill='white', linewidth=1.5, outlier.color = NA)+
    stat_compare_means(comparisons = comparison_list)+
    scale_color_manual(values = c('#e01a1a', '#878787', '#5b7aa2'))+
    scale_y_continuous(limits = y_lim)+
    scale_x_discrete(labels=x_labels)+
    labs(title=chrom, x='H2P prediction', y='Colocation ratio')+
    theme_bw()+
    theme(
      strip.background = element_rect(color=NA, fill='white', linewidth=1, linetype = 'solid'),
      strip.text = element_text(size=12, color='black'),
      panel.spacing = unit(1, 'cm'),
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 14, vjust = 0),
      axis.text = element_text(size=12, color='black'),
      axis.text.x = element_text(size=12, color='black', angle = 0),
      axis.title = element_text(size=14),
      legend.position = 'none')
}


### 01.每条染色体的FISH
comparison_list <- list(c('Pos', 'Random'), c('Random', 'Neg'), c('Pos', 'Neg'))
### plot chr1.
df_chr1_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr1_120000000-140000000.txt', header = TRUE)
df_chr1_H2P_coloc_ratio$Group <- factor(df_chr1_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr1 <- plot_H2P_coloc_ratio_boxplot(df_chr1_H2P_coloc_ratio, chrom_region = 'chr1_120000000-140000000')

### plot chr2.
df_chr2_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr2_100000000-120000000.txt', header = TRUE)
df_chr2_H2P_coloc_ratio$Group <- factor(df_chr2_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr2 <- plot_H2P_coloc_ratio_boxplot(df_chr2_H2P_coloc_ratio, chrom_region = 'chr2_100000000-120000000')

### plot chr3.
df_chr3_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr3_0-20000000.txt', header = TRUE)
df_chr3_H2P_coloc_ratio$Group <- factor(df_chr3_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr3 <- plot_H2P_coloc_ratio_boxplot(df_chr3_H2P_coloc_ratio, chrom_region = 'chr3_0-20000000')

### plot chr4.
df_chr4_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr4_80000000-100000000.txt', header = TRUE)
df_chr4_H2P_coloc_ratio$Group <- factor(df_chr4_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr4 <- plot_H2P_coloc_ratio_boxplot(df_chr4_H2P_coloc_ratio, chrom_region = 'chr4_80000000-100000000')

### plot chr5.
df_chr5_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr5_120000000-140000000.txt', header = TRUE)
df_chr5_H2P_coloc_ratio$Group <- factor(df_chr5_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr5 <- plot_H2P_coloc_ratio_boxplot(df_chr5_H2P_coloc_ratio, chrom_region = 'chr5_120000000-140000000')

### plot chr6.
df_chr6_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr6_40000000-60000000.txt', header = TRUE)
df_chr6_H2P_coloc_ratio$Group <- factor(df_chr6_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr6 <- plot_H2P_coloc_ratio_boxplot(df_chr6_H2P_coloc_ratio, chrom_region = 'chr6_40000000-60000000')

### plot chr8.
df_chr8_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr8_40000000-60000000.txt', header = TRUE)
df_chr8_H2P_coloc_ratio$Group <- factor(df_chr8_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr8 <- plot_H2P_coloc_ratio_boxplot(df_chr8_H2P_coloc_ratio, chrom_region = 'chr8_40000000-60000000')

### plot chr9.
df_chr9_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr9_100000000-120000000.txt', header = TRUE)
df_chr9_H2P_coloc_ratio$Group <- factor(df_chr9_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr9 <- plot_H2P_coloc_ratio_boxplot(df_chr9_H2P_coloc_ratio, chrom_region = 'chr9_100000000-120000000')

### plot chr10.
df_chr10_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr10_60000000-80000000.txt', header = TRUE)
df_chr10_H2P_coloc_ratio$Group <- factor(df_chr10_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr10 <- plot_H2P_coloc_ratio_boxplot(df_chr10_H2P_coloc_ratio, chrom_region = 'chr10_60000000-80000000')

### plot chr11.
df_chr11_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr11_80000000-100000000.txt', header = TRUE)
df_chr11_H2P_coloc_ratio$Group <- factor(df_chr11_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr11 <- plot_H2P_coloc_ratio_boxplot(df_chr11_H2P_coloc_ratio, chrom_region = 'chr11_80000000-100000000')

### plot chr12.
df_chr12_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr12_60000000-80000000.txt', header = TRUE)
df_chr12_H2P_coloc_ratio$Group <- factor(df_chr12_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr12 <- plot_H2P_coloc_ratio_boxplot(df_chr12_H2P_coloc_ratio, chrom_region = 'chr12_60000000-80000000')

### plot chr13.
df_chr13_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr13_20000000-40000000.txt', header = TRUE)
df_chr13_H2P_coloc_ratio$Group <- factor(df_chr13_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr13 <- plot_H2P_coloc_ratio_boxplot(df_chr13_H2P_coloc_ratio, chrom_region = 'chr13_20000000-40000000')

### plot chr14.
df_chr14_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr14_60000000-80000000.txt', header = TRUE)
df_chr14_H2P_coloc_ratio$Group <- factor(df_chr14_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr14 <- plot_H2P_coloc_ratio_boxplot(df_chr14_H2P_coloc_ratio, chrom_region = 'chr14_60000000-80000000')

### plot chr15.
df_chr15_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr15_60000000-80000000.txt', header = TRUE)
df_chr15_H2P_coloc_ratio$Group <- factor(df_chr15_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr15 <- plot_H2P_coloc_ratio_boxplot(df_chr15_H2P_coloc_ratio, chrom_region = 'chr15_60000000-80000000')

### plot chr16.
df_chr16_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr16_80000000-98207768.txt', header = TRUE)
df_chr16_H2P_coloc_ratio$Group <- factor(df_chr16_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr16 <- plot_H2P_coloc_ratio_boxplot(df_chr16_H2P_coloc_ratio, chrom_region = 'chr16_chr16_80000000-98207768')

### plot chr17.
df_chr17_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr17_60000000-80000000.txt', header = TRUE)
df_chr17_H2P_coloc_ratio$Group <- factor(df_chr17_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr17 <- plot_H2P_coloc_ratio_boxplot(df_chr17_H2P_coloc_ratio, chrom_region = 'chr17_60000000-80000000')

### plot chr18.
df_chr18_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr18_60000000-80000000.txt', header = TRUE)
df_chr18_H2P_coloc_ratio$Group <- factor(df_chr18_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr18 <- plot_H2P_coloc_ratio_boxplot(df_chr18_H2P_coloc_ratio, chrom_region = 'chr18_60000000-80000000')

### plot chr19.
df_chr19_H2P_coloc_ratio <- read.table('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_chr19_0-20000000.txt', header = TRUE)
df_chr19_H2P_coloc_ratio$Group <- factor(df_chr19_H2P_coloc_ratio$Group, levels = c('Pos', 'Random', 'Neg'))
p_chr19 <- plot_H2P_coloc_ratio_boxplot(df_chr19_H2P_coloc_ratio, chrom_region = 'chr19_0-20000000')

pdf('H2P_Pred_Coloc_ratio_within_150nm_chrom_boxplot.pdf', width = 4*5, height = 4*3.5)
ggarrange(p_chr1, p_chr2, p_chr3, p_chr4, p_chr5,
          p_chr6, p_chr8, p_chr9, p_chr10, p_chr11,
          p_chr12, p_chr13, p_chr14, p_chr15, p_chr16,
          p_chr17, p_chr18, p_chr19,
          ncol = 5, nrow = 4)
dev.off()

### 02.所有染色体的FISH
df_H2P_coloc_ratio <- data.frame(Group=character(), Coloc_Ratio=numeric())
for (chrom_region in chrom_regions){
  cat(chrom_region, sep = '\n')
  df_chrom_H2P_coloc_ratio <- read.table(paste0('./H2P_Pred_Coloc_ratio/H2P_Pred_Coloc_ratio_within_150nm_', chrom_region, '.txt'), header = TRUE)
  df_H2P_coloc_ratio <- rbind(df_H2P_coloc_ratio, df_chrom_H2P_coloc_ratio)
}

pdf('H2P_Pred_Coloc_ratio_within_150nm_mESC_wholeGenome_boxplot.pdf', width=4, height = 4)
plot_H2P_coloc_ratio_boxplot(df_H2P_coloc_ratio, chrom_region = 'mESC_whole_genome')
dev.off()
