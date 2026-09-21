setwd('D:/Project/HiC2PoreC/Analysis_Results/2026-01-21_RNAseq_PCA')

library(biomaRt)
library(tidyverse)
library(purrr)
library(dplyr)
library(DOSE)
library(ggstatsplot)
# library(pcaMethods)
library(factoextra)
library(FactoMineR)
library(ggrepel)
library(pheatmap)
library(ComplexHeatmap)
library(clusterProfiler)
library(org.Hs.eg.db)
library(org.Mm.eg.db)
library(ggplot2)
library(ggpubr)
library(ggforce)
library(circlize)

############# 1. 不同物种之间的ortholog. ################
## 查看可用物种
mart <- useMart('ensembl')
listDatasets(mart)

### 构建物种mart
human <- useMart(biomart = "ENSEMBL_MART_ENSEMBL", dataset = "hsapiens_gene_ensembl", host = 'https://dec2021.archive.ensembl.org/') ## GRCh38.p14
mouse <- useMart(biomart = "ENSEMBL_MART_ENSEMBL", dataset = "mmusculus_gene_ensembl", host = 'https://dec2021.archive.ensembl.org/') ## GRCm39
# mouse <- useEnsembl(biomart = 'genes', dataset = 'mmusculus_gene_ensembl', version = '102') ## GRCm38/mm10

### 读取物种基因
read_gene_bed <- function(gene_bed_file){
  df_gene <- read.table(gene_bed_file, header = FALSE, sep = '\t')
  names(df_gene) <- c('chrom', 'start', 'end', 'gene_name', 'gene_len', 'strand')
  
  return(df_gene)
  
}

df_hg38_gene <- read_gene_bed('hg38.gene.bed')
df_mm10_gene <- read_gene_bed('mm10.gene.bed')


m2h <- getLDS(attributes = c("mgi_symbol"),filters = "mgi_symbol",
                values=df_mm10_gene$gene_name, mart = mouse,
                attributesL = c("hgnc_symbol","chromosome_name","start_position"),
                martL = human, uniqueRows = T)


df_m2h_ortholog <- data.frame(m_gene=m2h$MGI.symbol, h_gene=m2h$HGNC.symbol)
df_m2h_ortholog <- df_m2h_ortholog[df_m2h_ortholog$h_gene != '',] ## 20,891个同源基因

df_ortholog <- purrr::reduce(list(df_m2h_ortholog), inner_join, by = "h_gene") ## 11,168个同源基因


############# 2. 几个细胞系的基因表达情况: GM12878/K562/hESC/mESC. ################
df_hg38_gene_Name2ID <- bitr(df_hg38_gene$gene_name, fromType = "SYMBOL", toType = "ENSEMBL", OrgDb = org.Hs.eg.db)
names(df_hg38_gene_Name2ID) <- c('gene_name', 'gene_id')
df_mm10_gene_Name2ID <- bitr(df_mm10_gene$gene_name, fromType = "SYMBOL", toType = "ENSEMBL", OrgDb = org.Mm.eg.db) 
names(df_mm10_gene_Name2ID) <- c('gene_name', 'gene_id')

bitr(df_exp$gene_id, fromType = "ENTREZID", toType = "SYMBOL", OrgDb = org.Mm.eg.db)

simplify_sample_gene_exp <- function(gene_exp_file, genome='hg38', df_gene_Name2ID=NULL, metrics='TPM', is_norm=TRUE){
  
  if (genome %in% c('hg38', 'mm10')){
    sample_name <- strsplit(basename(gene_exp_file), '\\.')[[1]][1]
    df_exp <- read.table(gene_exp_file, sep='\t', header = T)
    df_exp$gene_id <- sub("\\..*", "", df_exp$gene_id)
    df_exp <- df_exp[, c("gene_id", "TPM", "FPKM")]
    df_exp <- merge(df_gene_Name2ID, df_exp, by = 'gene_id')
    df_exp <- df_exp[, c('gene_name', metrics)]
    names(df_exp) <- c('gene_name', sample_name)
  } else{
    sample_name <- strsplit(basename(rsem_gene_exp_file), '\\.')[[1]][1]
    df_exp <- read.table(rsem_gene_exp_file, sep='\t', header = T)
    df_exp <- df_exp[, c("gene_id", "TPM", "FPKM")]
    df_exp <- df_exp[, c('gene_id', metrics)]
    names(df_exp) <- c('gene_name', sample_name)
  }
  
  dup_genes <- names(table(df_exp$gene_name)[table(df_exp$gene_name) > 1])
  
  df_exp <- df_exp[!df_exp$gene_name %in% dup_genes, ]
  
  if (is_norm){
    df_exp[, sample_name] = df_exp[, sample_name] / mean(df_exp[, sample_name])
  }
  
  return(df_exp)
}

df_GM12878_rep1_exp <- simplify_sample_gene_exp('GM12878_rep1.hg38_ENCFF910XWA.RNAseq.tsv', genome = 'hg38', df_gene_Name2ID = df_hg38_gene_Name2ID, metrics = 'TPM', is_norm = TRUE)
df_GM12878_rep2_exp <- simplify_sample_gene_exp('GM12878_rep2.hg38_ENCFF413MYB.RNAseq.tsv', genome = 'hg38', df_gene_Name2ID = df_hg38_gene_Name2ID, metrics = 'TPM', is_norm = TRUE)
df_K562_rep1_exp <- simplify_sample_gene_exp('K562_rep1.hg38_ENCFF384BFE.RNAseq.tsv', genome = 'hg38', df_gene_Name2ID = df_hg38_gene_Name2ID, metrics = 'TPM', is_norm = TRUE)
df_K562_rep2_exp <- simplify_sample_gene_exp('K562_rep2.hg38_ENCFF611MXW.RNAseq.tsv', genome = 'hg38', df_gene_Name2ID = df_hg38_gene_Name2ID, metrics = 'TPM', is_norm = TRUE)
df_hESC_rep1_exp <- simplify_sample_gene_exp('hESC_rep1.hg38_ENCFF216CFE.RNAseq.tsv', genome = 'hg38', df_gene_Name2ID = df_hg38_gene_Name2ID, metrics = 'TPM', is_norm = TRUE)
df_hESC_rep2_exp <- simplify_sample_gene_exp('hESC_rep2.hg38_ENCFF274WSK.RNAseq.tsv', genome = 'hg38', df_gene_Name2ID = df_hg38_gene_Name2ID, metrics = 'TPM', is_norm = TRUE)
df_mESC_rep1_exp <- simplify_sample_gene_exp('mESC_rep1.mm10_ENCFF827OZU.RNAseq.tsv', genome = 'mm10', df_gene_Name2ID = df_mm10_gene_Name2ID, metrics = 'TPM', is_norm = TRUE)
df_mESC_rep2_exp <- simplify_sample_gene_exp('mESC_rep2.mm10_ENCFF827OZU.RNAseq.tsv', genome = 'mm10', df_gene_Name2ID = df_mm10_gene_Name2ID, metrics = 'TPM', is_norm = TRUE)


df_human_exp <- purrr::reduce(list(df_GM12878_rep1_exp, df_GM12878_rep2_exp, df_K562_rep1_exp, df_K562_rep2_exp, df_hESC_rep1_exp, df_hESC_rep2_exp), inner_join, by = "gene_name")
names(df_human_exp)[1] <- 'h_gene'

df_mouse_exp <- purrr::reduce(list(df_mESC_rep1_exp, df_mESC_rep2_exp), inner_join, by = "gene_name")
names(df_mouse_exp)[1] <- 'm_gene'

############# 3. 2个物种gene ortholog表达水平, 做PCA. ################
df_ortholog

df_human_ortholog_exp <- df_human_exp[df_human_exp$h_gene %in% df_ortholog$h_gene, ]
df_mouse_ortholog_exp <- df_mouse_exp[df_mouse_exp$m_gene %in% df_ortholog$m_gene, ]

df_ortholog_exp <- merge(df_ortholog, df_human_ortholog_exp, by='h_gene')
df_ortholog_exp <- merge(df_ortholog_exp, df_mouse_ortholog_exp, by='m_gene')



df_ortholog_exp_cor <- cor(df_ortholog_exp[, 3:ncol(df_ortholog_exp)])

pheatmap::pheatmap(df_ortholog_exp_cor, cluster_rows = FALSE, cluster_cols = FALSE)

#### PCA.
df_ortholog_exp_pca <- df_ortholog_exp[, 2:ncol(df_ortholog_exp)]
df_ortholog_exp_pca <- aggregate(df_ortholog_exp_pca[, 2:ncol(df_ortholog_exp_pca)], by=list(df_ortholog_exp_pca$h_gene), mean)
names(df_ortholog_exp_pca)[1] <- 'h_gene'
# df_ortholog_exp_pca <- df_ortholog_exp_pca[! df_ortholog_exp_pca$h_gene %in% names(table(df_ortholog_exp_pca$h_gene)[table(df_ortholog_exp_pca$h_gene) > 1]), ]
rownames(df_ortholog_exp_pca) <- df_ortholog_exp_pca$h_gene
df_ortholog_exp_pca <- df_ortholog_exp_pca[, 2:ncol(df_ortholog_exp_pca)]

df_ortholog_exp_pca_t <- t(df_ortholog_exp_pca)
pca_res <- PCA(df_ortholog_exp_pca_t, graph = FALSE)

pca_res$ind$coord

df_pca_res <- data.frame(Sample=row.names(as.data.frame(pca_res$ind$coord)),
                         PC1=as.data.frame(pca_res$ind$coord)[, 1],
                         PC2=as.data.frame(pca_res$ind$coord)[, 2])

df_pca_res$Species <- c(rep('GM12878', 2),
                        rep('K562', 2),
                        rep('hESC', 2),
                        rep('mESC', 2))
df_pca_res$Species <- factor(df_pca_res$Species, levels = c('GM12878', 'K562', 'hESC', 'mESC'))

df_text <- df_pca_res[df_pca_res$Species %in% c('GM12878', 'K562', 'hESC', 'mESC'), ]
pdf('RNA-seq_exp_pca.pdf', height = 5, width = 6)
ggplot(data = df_pca_res, aes(x=PC1, y=PC2, color=Species))+
  # geom_point(aes(colour=Species), size=5))+
  geom_jitter(aes(colour=Species), size=5, alpha=0.7)+
  scale_color_manual(values = c('#c93835', '#f7ab60', '#80519f', '#072e8d'))+
  # scale_fill_manual(values = c(15, 16, 17, 18, 7, 12))+
  # stat_ellipse(aes(fill=Species), geom='polygon', alpha=0.2, linewidth=1, linetype=2)+
  # scale_x_continuous(limits = c(-100, 100))+
  # scale_y_continuous(limits = c(-40, 120))+
  geom_hline(yintercept = 0, linetype='dashed', linewidth=0.5)+
  geom_vline(xintercept = 0, linetype='dashed', linewidth=0.5)+
  # geom_text(label = df_pca_res$Sample)+
  labs(x='PC1 (38.5%)', y='PC2 (31.7%)', title='RNA-seq PCA')+
  geom_text_repel(data = df_text, aes(x=PC1, y=PC2, label=Sample))+
  theme_bw()+
  theme(
        axis.text = element_text(size=14, face="bold", color='black'),
        # axis.text.x = element_text(size=12, face='bold', color='black', angle=45),
        axis.title = element_text(size=16, face='bold', color='black'),
        # axis.line.x = element_line(size = 0.5, colour = "black"),
        #axis.line.y = element_line(size = 0.5, colour = "black"),
        legend.text = element_text(size=14, face="bold", color='black'),
        legend.title = element_blank(),
        plot.title = element_text(size=16, face='bold', color='black', hjust = 0.5)
  )
dev.off()

fviz_pca_ind(pca_res,
             geom.ind = "point", #c( "point", "text" ), # show points only (nbut not "text")
             col.ind = c(rep('human', 6),
                         rep('mouse', 2)), # color by groups
             palette = "Dark2",
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "Groups")+
  # ggtitle(this_title)+ 
  theme(plot.title = element_text(size=12,hjust = 0.5))

# 提取所有基因在PC1上的载荷
gene_loadings_pc1 <- pca_res$var$coord[, "Dim.1"] # "Dim.1"是PC1
loadings_abs <- abs(gene_loadings_pc1)
threshold <- quantile(loadings_abs, probs = 0.95) # 95%分位数: Top 5%
pc1_important_genes <- names(loadings_abs[loadings_abs >= threshold])

# 提取所有基因在PC2上的载荷
gene_loadings_pc2 <- pca_res$var$coord[, "Dim.2"] # "Dim.2"是PC2
loadings_abs <- abs(gene_loadings_pc2)
threshold <- quantile(loadings_abs, probs = 0.95) # 95%分位数: Top 5%
pc2_important_genes <- names(loadings_abs[loadings_abs >= threshold])

pca_importmant_genes <- c(pc1_important_genes, pc2_important_genes)

write.table(pc1_important_genes, 'NPCs.PC1_important_gene_list.txt', sep='\t', col.names = F, row.names = F, quote = F)
write.table(pc2_important_genes, 'NPCs.PC2_important_gene_list.txt', sep='\t', col.names = F, row.names = F, quote = F)

###### clusterprofiler注释不到，选择在线GO富集工具：ShinyGO
# pc1_gene_ego <- enrichGO(gene = pc1_important_genes, OrgDb = org.Hs.eg.db, ont = 'ALL', pAdjustMethod='BH', pvalueCutoff=0.05, qvalueCutoff=0.01)
# pc1_gene_ego <- as.data.frame(pc1_gene_ego)
# df_gm12878_go$FoldEnrichment <- parse_ratio(df_gm12878_go$GeneRatio)/parse_ratio(df_gm12878_go$BgRatio)
df_pc1_gene_goBP <- read.csv('NPCs.PC1_important_gene_list.GO_BP_enrichment_all.csv')
df_pc2_gene_goBP <- read.csv('NPCs.PC2_important_gene_list.GO_BP_enrichment_all.csv')

NPCs_pc1_genes <- df_pc1_gene_goBP %>% separate_rows(Genes, sep=' ') %>% pull(Genes) %>% unique()
NPCs_pc2_genes <- df_pc2_gene_goBP %>% separate_rows(Genes, sep=' ') %>% pull(Genes) %>% unique()

df_pc1_gene_exp <- df_ortholog_exp_pca[pc1_important_genes, ]
p1 <- Heatmap(df_pc1_gene_exp,
        cluster_columns = FALSE, cluster_rows = TRUE, show_row_names = FALSE)
p1 <- draw(p1)

df_pc2_gene_exp <- df_ortholog_exp_pca[pc2_important_genes, ]
p2 <- Heatmap(df_pc2_gene_exp,
        cluster_columns = FALSE, cluster_rows = TRUE, show_row_names = FALSE)
p2 <- draw(p2)

df_pca_important_gene_exp <- rbind(df_pc1_gene_exp[row_order(p1), ], df_pc2_gene_exp[row_order(p2), ])
df_pca_NPC_important_gene_exp <- rbind(df_pc1_gene_exp[row_order(p1), ], df_pc2_gene_exp[row_order(p2), ])

## 热图参考: https://www.jieandze1314.com/post/cnposts/195/
col_anno <- data.frame(Species=c(rep('human', 6),
                                 rep('chimp', 2),
                                 rep('gorilla', 3),
                                 rep('rhesus', 3),
                                 rep('dog', 2),
                                 rep('mouse', 3)))

colors = list(Species=c('human'='#e6ab02', 'chimp'='#389a76', 'gorilla'='#7085bf', 'rhesus'='#80519f', 'dog'='#b62835', 'mouse'='#8d8d8d'))

colAnn <- HeatmapAnnotation(df = col_anno, col = colors, which='col', na_col = 'white',
                            annotation_height = 0.6,
                            annotation_width = unit(1, 'cm'),
                            gap = unit(1, 'mm'),
                            annotation_legend_param = list(
                                Species = list(
                                  nrow = 1, # 这个legend显示几行
                                  title = 'Species',
                                  title_position = 'topcenter',
                                  legend_direction = 'vertical',
                                  title_gp = gpar(fontsize = 12, fontface = 'bold'),
                                  labels_gp = gpar(fontsize = 12, fontface = 'bold'))
                              ))

row_anno <- list('pc1' = paste(head(df_pc1_gene_goBP$Pathway, 20), collapse = '\n'), 
                 'pc2' = paste(head(df_pc2_gene_goBP$Pathway, 20), collapse = '\n'))

rowAnn <- rowAnnotation(GO_Terms = anno_empty(
                                          border = FALSE,
                                          width = max_text_width(unlist(row_anno)) * 1.2
                                        ),
                                        show_annotation_name = TRUE,
                                        annotation_name_side = "top",
                                        annotation_name_rot = 0,
                                        annotation_name_gp = gpar(fontsize = 10, fontface = "bold"))

# myBreaks <- seq(-3, 3, length.out = 100)
# myCol <- colorRampPalette(c('dodgerblue', 'black', 'yellow'))(100)
# col = colorRamp2(myBreaks, myCol)

p <- Heatmap(df_pca_important_gene_exp, name='Expression',
        cluster_rows = F, cluster_columns = F,
        col = colorRamp2(c(0, 5, 20), c('#075293', 'black', 'yellow')), # c('#0000ff', '#eae7ef', '#ff0000')
        # col = colorRamp2(c(0, 10, 20), c('#010364', '#f7f7ff', '#880100')), # c('#0000ff', '#eae7ef', '#ff0000')
        # col = colorRamp2(c(0, 15), c('#f7f7ff', '#880100')),
        row_title_side = 'left',
        row_title_gp = gpar(fontsize = 12,  fontface = 'bold'),
        row_title_rot = 90,
        show_row_names = F,
        show_row_dend = F,
        row_names_gp = gpar(fontsize = 10, fontface = 'bold'),
        row_names_side = 'left',
        row_dend_width = unit(25,'mm'),
        
        border = T,
        
        show_column_dend = F,
        show_column_names = TRUE,
        column_title = 'NPCs total RNA',
        column_title_side = 'top',
        column_title_gp = gpar(fontsize = 15, fontface = 'bold'),
        column_title_rot = 0,
        column_names_gp = gpar(fontsize = 10, fontface = 'bold'),
        column_names_max_height = unit(10, 'cm'),
        column_dend_height = unit(25,'mm'),
        column_names_rot = 60,
        
        row_split = c(rep('PC1 Top 5% genes', nrow(df_pc1_gene_exp)),
                      rep('PC2 Top 5% genes', nrow(df_pc2_gene_exp))),
        column_split = factor(c(rep('human', 6),
                                rep('chimp', 2),
                                rep('gorilla', 3),
                                rep('rhesus', 3),
                                rep('dog', 2),
                                rep('mouse', 3)), levels = c('human', 'chimp', 'gorilla', 'rhesus', 'dog', 'mouse')),
        top_annotation = colAnn,
        right_annotation = rowAnn,
        heatmap_legend_param = list(
          color_bar = 'continuous',
          legend_direction = 'horizontal', #vertical
          legend_width = unit(10, 'cm'),
          legend_height = unit(3, 'cm'),
          gap = unit(3, "cm"),
          title_position = 'topcenter',
          title_gp=gpar(fontsize = 12, fontface = 'bold'),
          labels_gp=gpar(fontsize = 12, fontface = 'bold')))

pdf('NPCs.PCA_gene_exp_GO.pdf', height = 12, width=15)
draw(p, heatmap_legend_side = 'bottom', annotation_legend_side = 'bottom')
for(i in 1:2) {
  decorate_annotation("GO_Terms", slice = i, {
    grid.text(row_anno[[paste0('pc', i)]], 
              x = 0, y = 0.5, just = "left",
              gp = gpar(fontsize = 12, fontface='bold', lineheight = 1))
  })
}
dev.off()

##### GO BP top 20 pathways: gene expression boxplot. 
plot_NPC_boxplot <- function(data=df_single_gene_exp, title_name=gene_name){ 
  p <- ggplot(data = df_single_gene_exp, aes(x=Species, y=Exp, fill=Species))+
    geom_boxplot()+
    # geom_point()+
    scale_fill_manual(values = c('#e6ab02', '#389a76', '#7085bf', '#80519f', '#b62835', '#8d8d8d'))+
    # scale_y_continuous(expand  = c(0,0))+
    labs(x='', y='Normalized TPM', title = title_name)+
    theme_bw(base_size = 14)+
    theme(
      strip.background = element_blank(),
      plot.title = element_text(hjust = 0.5, size=18),
      legend.position = 'none',
      axis.text = element_text(color = 'black'),
      axis.title = element_text(angle = 0, hjust = 0.5, vjust = 0.5, size=16, face='bold'),
      axis.text.x = element_text(angle = 30, hjust = 0.5, vjust = 0.5, size=14, face='bold'),
      axis.text.y = element_text(angle = 0, hjust = 0.5, vjust = 0.5, size=14, face='bold'))
  
  return(p)
}

########## NPC PC1 genes within top 20 BP 
for (i in seq(1, 20)){
  df_BP <- df_pc1_gene_goBP[i, ]
  pathway <- df_BP[, 'Pathway']
  genes <- unique(strsplit(df_BP[, 'Genes'], split = ' ')[[1]])
  
  cat(i, pathway, '\n')
  
  pathway <- paste(strsplit(pathway, ' |:')[[1]], collapse = '_')
  pdf_file <- paste0('./GO_BP_gene_expression_plot/NPCs_PC1_genes_within_Top20_BP/', pathway, '.gene_exp.pdf')  
  nrow <- ceiling(length(genes) / 6)
  ncol <- 6
  
  pdf(pdf_file, height = nrow*3.5, width = ncol*3)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(nrow, ncol)))
  a = 0
  for (gene in genes){
    
    if (gene != ''){
      
      df_single_gene_exp <- df_pca_important_gene_exp[gene, ]
      df_single_gene_exp <- reshape2::melt(df_single_gene_exp, variable.name='Sample', value.name='Exp')
      df_single_gene_exp$Species <- factor(c(rep('human', 6),
                                             rep('chimp', 2),
                                             rep('gorilla', 3),
                                             rep('rhesus', 3),
                                             rep('dog', 4),
                                             rep('mouse', 3)), levels = c('human', 'chimp', 'gorilla', 'rhesus', 'dog', 'mouse'))
      
      p <- plot_NPC_boxplot(data = df_single_gene_exp, title_name = gene)
      a = a + 1
      row = ifelse(a%%6 == 0, (a-1)%/%6+1, a%/%6+1)
      col = ifelse(a%%6 == 0, 6, a%%6)
      print(p, vp=viewport(layout.pos.row = row, layout.pos.col = col))
    }
  }
  dev.off()
  
}

########## NPC PC2 genes within top 20 BP 
for (i in seq(1, 20)){
  df_BP <- df_pc2_gene_goBP[i, ]
  pathway <- df_BP[, 'Pathway']
  genes <- unique(strsplit(df_BP[, 'Genes'], split = ' ')[[1]])
  
  cat(i, pathway, '\n')
  
  pathway <- paste(strsplit(pathway, ' |:')[[1]], collapse = '_')
  pdf_file <- paste0('./GO_BP_gene_expression_plot/NPCs_PC2_genes_within_Top20_BP/', pathway, '.gene_exp.pdf')  
  nrow <- ceiling(length(genes) / 6)
  ncol <- 6
  
  pdf(pdf_file, height = nrow*3.5, width = ncol*3)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(nrow, ncol)))
  a = 0
  for (gene in genes){
    
    if (gene != ''){
      
      df_single_gene_exp <- df_pca_important_gene_exp[gene, ]
      df_single_gene_exp <- reshape2::melt(df_single_gene_exp, variable.name='Sample', value.name='Exp')
      df_single_gene_exp$Species <- factor(c(rep('human', 6),
                                             rep('chimp', 2),
                                             rep('gorilla', 3),
                                             rep('rhesus', 3),
                                             rep('dog', 4),
                                             rep('mouse', 3)), levels = c('human', 'chimp', 'gorilla', 'rhesus', 'dog', 'mouse'))
      
      p <- plot_boxplot(data = df_single_gene_exp, title_name = gene)
      a = a + 1
      row = ifelse(a%%6 == 0, (a-1)%/%6+1, a%/%6+1)
      col = ifelse(a%%6 == 0, 6, a%%6)
      print(p, vp=viewport(layout.pos.row = row, layout.pos.col = col))
    }
  }
  dev.off()
  
}

############### 4. NeuN
##### 4.1 human NeuN
human_NeuN_samples <- c('human_NeuN_BB1_DLPFC','human_NeuN_BB1_OFC','human_NeuN_BB1_VLPFC',
                        'human_NeuN_BB2_DLPFC', 'human_NeuN_BB2_OFC', 'human_NeuN_BB2_VLPFC',
                        'human_NeuN_BB3_DLPFC', 'human_NeuN_BB3_OFC', 'human_NeuN_BB3_VLPFC',
                        'human_NeuN_BB4_DLPFC', 'human_NeuN_BB4_OFC', 'human_NeuN_BB4_VLPFC',
                        'human_NeuN_BB5_DLPFC', 'human_NeuN_BB5_OFC', 'human_NeuN_BB5_VLPFC',
                        'human_NeuN_BB6_DLPFC', 'human_NeuN_BB6_OFC', 'human_NeuN_BB6_VLPFC')

df_human_NeuN_BB1_DLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB1_DLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB2_DLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB2_DLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB3_DLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB3_DLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB4_DLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB4_DLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB5_DLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB5_DLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB6_DLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB6_DLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB1_OFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB1_OFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB2_OFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB2_OFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB3_OFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB3_OFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB4_OFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB4_OFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB5_OFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB5_OFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB6_OFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB6_OFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB1_VLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB1_VLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB2_VLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB2_VLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB3_VLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB3_VLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB4_VLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB4_VLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB5_VLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB5_VLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_human_NeuN_BB6_VLPFC_exp <- simplify_sample_gene_exp('species_gene_expression/human_NeuN_BB6_VLPFC.genes.results', genome = 'hg38', df_gene_ID2Name = df_hg38_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)

df_human_NeuN_exp <- purrr::reduce(list(df_human_NeuN_BB1_DLPFC_exp, df_human_NeuN_BB1_OFC_exp, df_human_NeuN_BB1_VLPFC_exp,
                                df_human_NeuN_BB2_DLPFC_exp, df_human_NeuN_BB2_OFC_exp, df_human_NeuN_BB2_VLPFC_exp,
                                df_human_NeuN_BB3_DLPFC_exp, df_human_NeuN_BB3_OFC_exp, df_human_NeuN_BB3_VLPFC_exp,
                                df_human_NeuN_BB4_DLPFC_exp, df_human_NeuN_BB4_OFC_exp, df_human_NeuN_BB4_VLPFC_exp,
                                df_human_NeuN_BB5_DLPFC_exp, df_human_NeuN_BB5_OFC_exp, df_human_NeuN_BB5_VLPFC_exp,
                                df_human_NeuN_BB6_DLPFC_exp, df_human_NeuN_BB6_OFC_exp, df_human_NeuN_BB6_VLPFC_exp), inner_join, by = "gene_name")
names(df_human_NeuN_exp)[1] <- 'h_gene'

##### 4.2 dog NeuN (Nuclear)
# dog_NeuN_samples <- c('AD', 'P0',
#                       'QS_A1', 'QS_A2', 'QS_A3', 'QS_A4', 'QS_A5', 'QS_A6',
#                       'QS_P01', 'QS_P02', 'QS_P03', 'QS_P04', 'QS_P05', 'QS_P06',
#                       'QS_RS1', 'QS_RS2', 'QS_RS3', 'QS_RS4', 'QS_RS5', 'QS_RS6')
# 
# df_dog_AD_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/AD-Neu-Nuclear-RNA-Rep1.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_P0_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/P0-Neu-Nuclear-RNA-Rep1.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_A1_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_A-1.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_A2_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_A-2.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_A3_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_A-3.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_A4_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_A-4.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_A5_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_A-5.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_A6_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_A-6.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_P01_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_P0-1.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_P02_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_P0-2.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_P03_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_P0-3.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_P04_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_P0-4.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_P05_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_P0-5.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_P06_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_P0-6.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_RS1_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_RS-1.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_RS2_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_RS-2.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_RS3_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_RS-3.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_RS4_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_RS-4.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_RS5_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_RS-5.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_QS_RS6_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/QS_RS-6.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)

# df_dog_NeuN_exp <- purrr::reduce(list(df_dog_AD_NeuN_exp, df_dog_P0_NeuN_exp,
#                                       df_dog_QS_A1_NeuN_exp, df_dog_QS_A2_NeuN_exp, df_dog_QS_A3_NeuN_exp, df_dog_QS_A4_NeuN_exp, df_dog_QS_A5_NeuN_exp, df_dog_QS_A6_NeuN_exp,
#                                       df_dog_QS_P01_NeuN_exp, df_dog_QS_P02_NeuN_exp, df_dog_QS_P03_NeuN_exp, df_dog_QS_P04_NeuN_exp, df_dog_QS_P05_NeuN_exp, df_dog_QS_P06_NeuN_exp, 
#                                       df_dog_QS_RS1_NeuN_exp, df_dog_QS_RS2_NeuN_exp, df_dog_QS_RS3_NeuN_exp, df_dog_QS_RS4_NeuN_exp, df_dog_QS_RS5_NeuN_exp, df_dog_QS_RS6_NeuN_exp), inner_join, by = "gene_name")

# df_dog_P0_Rep1_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/P0_Nuclear_RNAseq_Rep1.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
# df_dog_P0_Rep2_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/P0_Nuclear_RNAseq_Rep2.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
df_dog_AD_Rep1_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/AD_Nuclear_RNAseq_Rep1.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)
df_dog_AD_Rep2_NeuN_exp <- simplify_sample_gene_exp('species_gene_expression/AD_Nuclear_RNAseq_Rep2.genes.results', genome = 'cfam1', df_gene_ID2Name = NULL, metrics = 'TPM', is_norm = TRUE)

df_dog_NeuN_exp <- purrr::reduce(list(df_dog_AD_Rep1_NeuN_exp, df_dog_AD_Rep2_NeuN_exp), inner_join, by = "gene_name")
names(df_dog_NeuN_exp)[1] <- 'd_gene'

###### 4.3 mouse NeuN
# mouse_NeuN_samples <- c('QS_M1', 'QS_M2', 'QS_M3', 'QS_M4')
# 
# # df_mouse_NeuN_12weeks_VEH1_1_exp <- simplify_sample_gene_exp('species_gene_expression/mouse_NeuN_12weeks_VEH1_1.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# # df_mouse_NeuN_12weeks_VEH1_2_exp <- simplify_sample_gene_exp('species_gene_expression/mouse_NeuN_12weeks_VEH1_2.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# # df_mouse_NeuN_12weeks_VEH2_1_exp <- simplify_sample_gene_exp('species_gene_expression/mouse_NeuN_12weeks_VEH2_1.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# # df_mouse_NeuN_12weeks_VEH2_2_exp <- simplify_sample_gene_exp('species_gene_expression/mouse_NeuN_12weeks_VEH2_2.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# # df_mouse_NeuN_12weeks_VEH3_1_exp <- simplify_sample_gene_exp('species_gene_expression/mouse_NeuN_12weeks_VEH3_1.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# # df_mouse_NeuN_12weeks_VEH3_2_exp <- simplify_sample_gene_exp('species_gene_expression/mouse_NeuN_12weeks_VEH3_2.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# # df_mouse_NeuN_12weeks_VEH4_1_exp <- simplify_sample_gene_exp('species_gene_expression/mouse_NeuN_12weeks_VEH4_1.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# # df_mouse_NeuN_12weeks_VEH4_2_exp <- simplify_sample_gene_exp('species_gene_expression/mouse_NeuN_12weeks_VEH4_2.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# 
# df_mouse_QS_M1_NeuN <- simplify_sample_gene_exp('species_gene_expression/QS_M-1.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# df_mouse_QS_M2_NeuN <- simplify_sample_gene_exp('species_gene_expression/QS_M-2.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# df_mouse_QS_M3_NeuN <- simplify_sample_gene_exp('species_gene_expression/QS_M-3.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
# df_mouse_QS_M4_NeuN <- simplify_sample_gene_exp('species_gene_expression/QS_M-4.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)

# df_mouse_NeuN_exp <- purrr::reduce(list(df_mouse_NeuN_12weeks_VEH1_1_exp, df_mouse_NeuN_12weeks_VEH1_2_exp,
#                                  df_mouse_NeuN_12weeks_VEH2_1_exp, df_mouse_NeuN_12weeks_VEH2_2_exp,
#                                  df_mouse_NeuN_12weeks_VEH3_1_exp, df_mouse_NeuN_12weeks_VEH3_2_exp,
#                                  df_mouse_NeuN_12weeks_VEH4_1_exp, df_mouse_NeuN_12weeks_VEH4_2_exp,
#                                  df_mouse_QS_M1_NeuN, df_mouse_QS_M2_NeuN, df_mouse_QS_M3_NeuN, df_mouse_QS_M4_NeuN), inner_join, by = "gene_name")

df_mouse_NeuN_Rep1_exp <- simplify_sample_gene_exp('species_gene_expression/Mouse_NeuN_Nuclear_RNAseq_Rep1.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)
df_mouse_NeuN_Rep2_exp <- simplify_sample_gene_exp('species_gene_expression/Mouse_NeuN_Nuclear_RNAseq_Rep2.genes.results', genome = 'mm10', df_gene_ID2Name = df_mm10_gene_ID2Name, metrics = 'TPM', is_norm = TRUE)

df_mouse_NeuN_exp <- purrr::reduce(list(df_mouse_NeuN_Rep1_exp, df_mouse_NeuN_Rep2_exp), inner_join, by = "gene_name")
names(df_mouse_NeuN_exp)[1] <- 'm_gene'

############# 4.4. 3个物种NeuN gene ortholog表达水平, 做PCA. ################
df_human_ortholog_exp <- df_human_NeuN_exp[df_human_NeuN_exp$h_gene %in% df_ortholog$h_gene, ]
df_dog_ortholog_exp <- df_dog_NeuN_exp[df_dog_NeuN_exp$d_gene %in% df_ortholog$d_gene, ]
df_mouse_ortholog_exp <- df_mouse_NeuN_exp[df_mouse_NeuN_exp$m_gene %in% df_ortholog$m_gene, ]

df_ortholog_exp <- merge(df_ortholog, df_human_ortholog_exp, by='h_gene')
df_ortholog_exp <- merge(df_ortholog_exp, df_dog_ortholog_exp, by='d_gene')
df_ortholog_exp <- merge(df_ortholog_exp, df_mouse_ortholog_exp, by='m_gene')

df_ortholog_exp_cor <- cor(df_ortholog_exp[, 7:ncol(df_ortholog_exp)])

pheatmap::pheatmap(df_ortholog_exp_cor, cluster_rows = FALSE, cluster_cols = FALSE)

#### PCA.
df_ortholog_exp_pca <- df_ortholog_exp[, c(3 ,7:ncol(df_ortholog_exp))]
df_ortholog_exp_pca <- aggregate(df_ortholog_exp_pca[, 2:ncol(df_ortholog_exp_pca)], by=list(df_ortholog_exp_pca$h_gene), mean)
names(df_ortholog_exp_pca)[1] <- 'h_gene'
# df_ortholog_exp_pca <- df_ortholog_exp_pca[! df_ortholog_exp_pca$h_gene %in% names(table(df_ortholog_exp_pca$h_gene)[table(df_ortholog_exp_pca$h_gene) > 1]), ]
rownames(df_ortholog_exp_pca) <- df_ortholog_exp_pca$h_gene
df_ortholog_exp_pca <- df_ortholog_exp_pca[, 2:ncol(df_ortholog_exp_pca)]

df_ortholog_exp_pca_t <- t(df_ortholog_exp_pca)
pca_res <- PCA(df_ortholog_exp_pca_t, graph = FALSE)

pca_res$ind$coord

df_pca_res <- data.frame(Sample=row.names(as.data.frame(pca_res$ind$coord)),
                         PC1=as.data.frame(pca_res$ind$coord)[, 1],
                         PC2=as.data.frame(pca_res$ind$coord)[, 2])

df_pca_res$Species <- c(rep('human', 18),
                        rep('dog', 2),
                        rep('mouse', 2))
df_pca_res$Species <- factor(df_pca_res$Species, levels = c('human', 'dog', 'mouse'))

df_text <- df_pca_res[df_pca_res$Species == 'dog', ]
pdf('Species_Neurons_ortholog_exp_pca.pdf', height = 5, width = 6)
ggplot(data = df_pca_res, aes(x=PC1, y=PC2, color=Species))+
  # geom_point(aes(colour=Species), size=5))+
  geom_jitter(aes(colour=Species), size=5, alpha=0.7)+
  scale_color_manual(values = c('#e6ab02', '#b62835', '#444b4d'))+
  geom_hline(yintercept = 0, linetype='dashed', linewidth=0.5)+
  geom_vline(xintercept = 0, linetype='dashed', linewidth=0.5)+
  # geom_text(label = df_pca_res$Sample)+
  labs(x='PC1 (48.5%)', y='PC2 (16.1%)', title='Neurons_Nuclear_RNA PCA')+
  geom_text_repel(data = df_text, aes(x=PC1, y=PC2, label=Sample))+
  theme_bw()+
  theme(
    axis.text = element_text(size=14, face="bold", color='black'),
    # axis.text.x = element_text(size=12, face='bold', color='black', angle=45),
    axis.title = element_text(size=16, face='bold', color='black'),
    # axis.line.x = element_line(size = 0.5, colour = "black"),
    #axis.line.y = element_line(size = 0.5, colour = "black"),
    legend.text = element_text(size=14, face="bold", color='black'),
    legend.title = element_blank(),
    plot.title = element_text(size=16, face='bold', color='black', hjust = 0.5)
  )
dev.off()

fviz_pca_ind(pca_res,
             geom.ind = "point", #c( "point", "text" ), # show points only (nbut not "text")
             col.ind = c(rep('human', 18),
                         rep('dog', 2),
                         rep('mouse', 2)), # color by groups
             palette = "Dark2",
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "Groups")+
  # ggtitle(this_title)+ 
  theme(plot.title = element_text(size=12,hjust = 0.5))


# 提取所有基因在PC1上的载荷
gene_loadings_pc1 <- pca_res$var$coord[, "Dim.1"] # "Dim.1"是PC1
loadings_abs <- abs(gene_loadings_pc1)
threshold <- quantile(loadings_abs, probs = 0.95) # 95%分位数: Top 5%
pc1_important_genes <- names(loadings_abs[loadings_abs >= threshold])

# 提取所有基因在PC2上的载荷
gene_loadings_pc2 <- pca_res$var$coord[, "Dim.2"] # "Dim.2"是PC2
loadings_abs <- abs(gene_loadings_pc2)
threshold <- quantile(loadings_abs, probs = 0.95) # 95%分位数: Top 5%
pc2_important_genes <- names(loadings_abs[loadings_abs >= threshold])

pca_importmant_genes <- c(pc1_important_genes, pc2_important_genes)

write.table(pc1_important_genes, 'Neurons.PC1_important_gene_list.txt', sep='\t', col.names = F, row.names = F, quote = F)
write.table(pc2_important_genes, 'Neurons.PC2_important_gene_list.txt', sep='\t', col.names = F, row.names = F, quote = F)

###### clusterprofiler注释不到，选择在线GO富集工具：ShinyGO
# pc1_gene_ego <- enrichGO(gene = pc1_important_genes, OrgDb = org.Hs.eg.db, ont = 'ALL', pAdjustMethod='BH', pvalueCutoff=0.05, qvalueCutoff=0.01)
# pc1_gene_ego <- as.data.frame(pc1_gene_ego)
# df_gm12878_go$FoldEnrichment <- parse_ratio(df_gm12878_go$GeneRatio)/parse_ratio(df_gm12878_go$BgRatio)
df_pc1_gene_goBP <- read.csv('Neurons.PC1_important_gene_list.GO_BP_enrichment_all.csv')
df_pc2_gene_goBP <- read.csv('Neurons.PC2_important_gene_list.GO_BP_enrichment_all.csv')

Neurons_pc1_genes <- df_pc1_gene_goBP %>% separate_rows(Genes, sep=' ') %>% pull(Genes) %>% unique()
Neurons_pc2_genes <- df_pc2_gene_goBP %>% separate_rows(Genes, sep=' ') %>% pull(Genes) %>% unique()

df_pc1_gene_exp <- df_ortholog_exp_pca[pc1_important_genes, ]
p1 <- Heatmap(df_pc1_gene_exp,
              cluster_columns = FALSE, cluster_rows = TRUE, show_row_names = FALSE)
p1 <- draw(p1)

df_pc2_gene_exp <- df_ortholog_exp_pca[pc2_important_genes, ]
p2 <- Heatmap(df_pc2_gene_exp,
              cluster_columns = FALSE, cluster_rows = TRUE, show_row_names = FALSE)
p2 <- draw(p2)

df_pca_important_gene_exp <- rbind(df_pc1_gene_exp[row_order(p1), ], df_pc2_gene_exp[row_order(p2), ])
df_pca_Neuron_important_gene_exp <- rbind(df_pc1_gene_exp[row_order(p1), ], df_pc2_gene_exp[row_order(p2), ])
## 热图参考: https://www.jieandze1314.com/post/cnposts/195/
col_anno <- data.frame(Species=c(rep('human', 18),
                                 rep('dog', 2),
                                 rep('mouse', 2)))

colors = list(Species=c('human'='#e6ab02', 'dog'='#b62835', 'mouse'='#8d8d8d'))

colAnn <- HeatmapAnnotation(df = col_anno, col = colors, which='col', na_col = 'white',
                            annotation_height = 0.6,
                            annotation_width = unit(1, 'cm'),
                            gap = unit(1, 'mm'),
                            annotation_legend_param = list(
                              Species = list(
                                nrow = 1, # 这个legend显示几行
                                title = 'Species',
                                title_position = 'topcenter',
                                legend_direction = 'vertical',
                                title_gp = gpar(fontsize = 12, fontface = 'bold'),
                                labels_gp = gpar(fontsize = 12, fontface = 'bold'))
                            ))

row_anno <- list('pc1' = paste(head(df_pc1_gene_goBP$Pathway, 20), collapse = '\n'), 
                 'pc2' = paste(head(df_pc2_gene_goBP$Pathway, 20), collapse = '\n'))

rowAnn <- rowAnnotation(GO_Terms = anno_empty(
                                border = FALSE,
                                width = max_text_width(unlist(row_anno)) * 1.2
                              ),
                              show_annotation_name = TRUE,
                              annotation_name_side = "top",
                              annotation_name_rot = 0,
                              annotation_name_gp = gpar(fontsize = 10, fontface = "bold"))

# myBreaks <- seq(-3, 3, length.out = 100)
# myCol <- colorRampPalette(c('dodgerblue', 'black', 'yellow'))(100)
# col = colorRamp2(myBreaks, myCol)

p <- Heatmap(df_pca_important_gene_exp, name='Expression',
             cluster_rows = F, cluster_columns = F,
             col = colorRamp2(c(0, 5, 20), c('#075293', 'black', 'yellow')), # c('#0000ff', '#eae7ef', '#ff0000')
             # col = colorRamp2(c(0, 20), c('#f7f7ff', '#880100')),
             row_title_side = 'left',
             row_title_gp = gpar(fontsize = 12,  fontface = 'bold'),
             row_title_rot = 90,
             show_row_names = F,
             show_row_dend = F,
             row_names_gp = gpar(fontsize = 10, fontface = 'bold'),
             row_names_side = 'left',
             row_dend_width = unit(25,'mm'),
             border = T,
             show_column_dend = F,
             show_column_names = TRUE,
             column_title = 'Neurons nuclear RNA',
             column_title_side = 'top',
             column_title_gp = gpar(fontsize = 15, fontface = 'bold'),
             column_title_rot = 0,
             column_names_gp = gpar(fontsize = 10, fontface = 'bold'),
             column_names_max_height = unit(10, 'cm'),
             column_dend_height = unit(25,'mm'),
             column_names_rot = 60,
             
             row_split = c(rep('PC1 Top 5% genes', nrow(df_pc1_gene_exp)),
                           rep('PC2 Top 5% genes', nrow(df_pc2_gene_exp))),
             column_split = factor(c(rep('human', 18),
                                     rep('dog', 2),
                                     rep('mouse', 2)), levels = c('human', 'dog', 'mouse')),
             top_annotation = colAnn,
             right_annotation = rowAnn,
             heatmap_legend_param = list(
               color_bar = 'continuous',
               legend_direction = 'horizontal', #vertical
               legend_width = unit(10, 'cm'),
               legend_height = unit(3, 'cm'),
               gap = unit(3, "cm"),
               title_position = 'topcenter',
               title_gp=gpar(fontsize = 12, fontface = 'bold'),
               labels_gp=gpar(fontsize = 12, fontface = 'bold')))

pdf('Neurons.PCA_gene_exp_GO.pdf', height = 14, width=15)
draw(p, heatmap_legend_side = 'bottom', annotation_legend_side = 'bottom')
for(i in 1:2) {
  decorate_annotation("GO_Terms", slice = i, {
    grid.text(row_anno[[paste0('pc', i)]], 
              x = 0, y = 0.5, just = "left",
              gp = gpar(fontsize = 12, fontface='bold', lineheight = 1))
  })
}
dev.off()

##########################
plot_Neuron_boxplot <- function(data=df_single_gene_exp, title_name=gene_name){ 
  p <- ggplot(data = df_single_gene_exp, aes(x=Species, y=Exp, fill=Species))+
    geom_boxplot()+
    # geom_point()+
    scale_fill_manual(values = c('#e6ab02', '#b62835', '#8d8d8d'))+
    # scale_y_continuous(expand  = c(0,0))+
    labs(x='', y='Normalized TPM', title = title_name)+
    theme_bw(base_size = 14)+
    theme(
      strip.background = element_blank(),
      plot.title = element_text(hjust = 0.5, size=18),
      legend.position = 'none',
      axis.text = element_text(color = 'black'),
      axis.title = element_text(angle = 0, hjust = 0.5, vjust = 0.5, size=16, face='bold'),
      axis.text.x = element_text(angle = 30, hjust = 0.5, vjust = 0.5, size=14, face='bold'),
      axis.text.y = element_text(angle = 0, hjust = 0.5, vjust = 0.5, size=14, face='bold'))
  
  return(p)
}

########## Neuron PC1 genes within top 20 BP 
for (i in seq(1, 20)){
  df_BP <- df_pc1_gene_goBP[i, ]
  pathway <- df_BP[, 'Pathway']
  genes <- unique(strsplit(df_BP[, 'Genes'], split = ' ')[[1]])
  
  cat(i, pathway, '\n')
  
  pathway <- paste(strsplit(pathway, ' |:')[[1]], collapse = '_')
  pdf_file <- paste0('./GO_BP_gene_expression_plot/Neurons_PC1_genes_within_Top20_BP/', pathway, '.gene_exp.pdf')  
  nrow <- ceiling(length(genes) / 6)
  ncol <- 6
  
  pdf(pdf_file, height = nrow*3.5, width = ncol*3)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(nrow, ncol)))
  a = 0
  for (gene in genes){
    
    if (gene != ''){
      
      df_single_gene_exp <- df_pca_important_gene_exp[gene, ]
      df_single_gene_exp <- reshape2::melt(df_single_gene_exp, variable.name='Sample', value.name='Exp')
      df_single_gene_exp$Species <- factor(c(rep('human', 18),
                                             rep('dog', 4),
                                             rep('mouse', 2)), levels = c('human', 'dog', 'mouse'))
      
      p <- plot_Neuron_boxplot(data = df_single_gene_exp, title_name = gene)
      a = a + 1
      row = ifelse(a%%6 == 0, (a-1)%/%6+1, a%/%6+1)
      col = ifelse(a%%6 == 0, 6, a%%6)
      print(p, vp=viewport(layout.pos.row = row, layout.pos.col = col))
    }
  }
  dev.off()
  
}

########## Neuron PC2 genes within top 20 BP 
for (i in seq(1, 20)){
  df_BP <- df_pc2_gene_goBP[i, ]
  pathway <- df_BP[, 'Pathway']
  genes <- unique(strsplit(df_BP[, 'Genes'], split = ' ')[[1]])
  
  cat(i, pathway, '\n')
  
  pathway <- paste(strsplit(pathway, ' |:')[[1]], collapse = '_')
  pdf_file <- paste0('./GO_BP_gene_expression_plot/Neurons_PC2_genes_within_Top20_BP/', pathway, '.gene_exp.pdf')  
  nrow <- ceiling(length(genes) / 6)
  ncol <- 6
  
  pdf(pdf_file, height = nrow*3.5, width = ncol*3)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(nrow, ncol)))
  a = 0
  for (gene in genes){
    
    if (gene != ''){
      
      df_single_gene_exp <- df_pca_important_gene_exp[gene, ]
      df_single_gene_exp <- reshape2::melt(df_single_gene_exp, variable.name='Sample', value.name='Exp')
      df_single_gene_exp$Species <- factor(c(rep('human', 18),
                                             rep('dog', 4),
                                             rep('mouse', 2)), levels = c('human', 'dog', 'mouse'))
      p <- plot_boxplot(data = df_single_gene_exp, title_name = gene)
      a = a + 1
      row = ifelse(a%%6 == 0, (a-1)%/%6+1, a%/%6+1)
      col = ifelse(a%%6 == 0, 6, a%%6)
      print(p, vp=viewport(layout.pos.row = row, layout.pos.col = col))
    }
  }
  dev.off()
  
}

################# Neuron HAR gene targets overlap with DEGs.
df_Neuron_HAR_gene_target <- openxlsx::read.xlsx('Human_Neuron_HAR_target_genes/1-s2.0-S0092867425000364-mmc1.xlsx', sheet = 23)

Neuron_HAR_gene_targets <- df_Neuron_HAR_gene_target %>% separate_rows(Gene.Target, sep = ',') %>% pull(Gene.Target) %>% unique()

intersect(NPCs_pc1_genes, Neuron_HAR_gene_targets)
intersect(NPCs_pc2_genes, Neuron_HAR_gene_targets)
intersect(Neurons_pc1_genes, Neuron_HAR_gene_targets)
intersect(Neurons_pc2_genes, Neuron_HAR_gene_targets)

library(eulerr)
library(GeneOverlap)
library(gridExtra)
## function:
get_pval_oddratio_venn <- function(g1, g2, genome.size=19871){
  go.obj <- newGeneOverlap(g1, g2, genome.size = genome.size) ## genome.size参数为物种基因数量
  go.obj <- testGeneOverlap(go.obj)
  pval <- signif(go.obj@pval, 5)
  odds_ratio <- round(go.obj@odds.ratio, 4)
  # return(go.obj@odds.ratio)
  return(c("pval"=pval,
           'odds_ratio'=odds_ratio))
}

calc_perc <- function(g1, g2){
  intersect_num <- length(intersect(g1, g2))
  g1_excluNum <- length(g1)-intersect_num
  g2_excluNum <- length(g2)-intersect_num
  g1_excluPerc <- g1_excluNum/length(g1)
  g2_excluPerc <- g2_excluNum/length(g2)
  g1_intersectPerc <- intersect_num / length(g1)
  g2_intersectPerc <- intersect_num / length(g2)
  
  g1_excluPerc <- paste0(round(g1_excluPerc, 4) * 100, '%')
  g2_excluPerc <- paste0(round(g2_excluPerc, 4) * 100, '%')
  g1_intersectPerc <- paste0(round(g1_intersectPerc, 4) * 100, '%')
  g2_intersectPerc <- paste0(round(g2_intersectPerc, 4) * 100, '%')
  return(c("intersect_num" = intersect_num,
           "g1_intersectPerc" = g1_intersectPerc,
           "g2_intersectPerc" = g2_intersectPerc,
           "g1_excluNum" = g1_excluNum, 
           "g2_excluNum" = g2_excluNum,
           "g1_excluPerc" = g1_excluPerc,
           "g2_excluPerc" = g2_excluPerc))
}

################## eulerr: plot venn diagram
####### p1: NPCs_PC1 VS HAR target genes
res <- calc_perc(NPCs_pc1_genes, Neuron_HAR_gene_targets)
venn_res <- get_pval_oddratio_venn(NPCs_pc1_genes, Neuron_HAR_gene_targets)
combo <- c("NPCs_PC1" = as.integer(res['g1_excluNum']),
           "HAR" = as.integer(res['g2_excluNum']),
           "NPCs_PC1&HAR" = as.integer(res['intersect_num']))
p1 <- plot(euler(combo),
           fills = list(fill=c('#c4ceea', '#f7d1d0')),
           quantities = T,
           edges = list(col='black', alpha=1, lwd=2),
           main = list(label=c('NPCs PC1'), cex=1.8))

####### p2: NPCs_PC2 VS HAR target genes
res <- calc_perc(NPCs_pc2_genes, Neuron_HAR_gene_targets)
venn_res <- get_pval_oddratio_venn(NPCs_pc2_genes, Neuron_HAR_gene_targets)
combo <- c("NPCs_PC2" = as.integer(res['g1_excluNum']),
           "HAR" = as.integer(res['g2_excluNum']),
           "NPCs_PC2&HAR" = as.integer(res['intersect_num']))
p2 <- plot(euler(combo),
           fills = list(fill=c('#c4ceea', '#f7d1d0')),
           quantities = T,
           edges = list(col='black', alpha=1, lwd=2),
           main = list(label=c('NPCs PC2'), cex=1.8))

####### p3: Neurons_PC1 VS HAR target genes
res <- calc_perc(Neurons_pc1_genes, Neuron_HAR_gene_targets)
venn_res <- get_pval_oddratio_venn(Neurons_pc1_genes, Neuron_HAR_gene_targets)
combo <- c("Neurons_PC1" = as.integer(res['g1_excluNum']),
           "HAR" = as.integer(res['g2_excluNum']),
           "Neurons_PC1&HAR" = as.integer(res['intersect_num']))
p3 <- plot(euler(combo),
           fills = list(fill=c('#c4ceea', '#f7d1d0')),
           quantities = T,
           edges = list(col='black', alpha=1, lwd=2),
           main = list(label=c('Neurons PC1'), cex=1.8))

####### p4: Neurons_PC1 VS HAR target genes
res <- calc_perc(Neurons_pc2_genes, Neuron_HAR_gene_targets)
venn_res <- get_pval_oddratio_venn(Neurons_pc2_genes, Neuron_HAR_gene_targets)
combo <- c("Neurons_PC2" = as.integer(res['g1_excluNum']),
           "HAR" = as.integer(res['g2_excluNum']),
           "Neurons_PC2&HAR" = as.integer(res['intersect_num']))
p4 <- plot(euler(combo),
           fills = list(fill=c('#c4ceea', '#f7d1d0')),
           quantities = T,
           edges = list(col='black', alpha=1, lwd=2),
           main = list(label=c('Neurons PC2'), cex=1.8))

pdf('Human_Neuron_HAR_target_genes/eulerr_venn_plot.pdf', height = 8, width = 8)
grid.arrange(p1, p2, p3, p4, ncol = 2)
dev.off()

#### PC2上与Neuron_HAR_targets gene有overlap的基因表达情况
############ 1. NPC
genes <- intersect(NPCs_pc2_genes, Neuron_HAR_gene_targets)
pdf_file <- paste0('./PC2_gene_overlap_HAR_targets_gene_expression/NPCs_PC2_OV_NeuronHARTarget.gene_exp.pdf')  
nrow <- ceiling(length(genes) / 5)
ncol <- 5

pdf(pdf_file, height = nrow*3.5, width = ncol*3)
grid.newpage()
pushViewport(viewport(layout = grid.layout(nrow, ncol)))
a = 0
for (gene in genes){
  
  if (gene != ''){
    
    df_single_gene_exp <- df_pca_NPC_important_gene_exp[gene, ]
    df_single_gene_exp <- reshape2::melt(df_single_gene_exp, variable.name='Sample', value.name='Exp')
    df_single_gene_exp$Species <- factor(c(rep('human', 6),
                                           rep('chimp', 2),
                                           rep('gorilla', 3),
                                           rep('rhesus', 3),
                                           rep('dog', 4),
                                           rep('mouse', 3)), levels = c('human', 'chimp', 'gorilla', 'rhesus', 'dog', 'mouse'))
    p <- plot_NPC_boxplot(data = df_single_gene_exp, title_name = gene)
    a = a + 1
    row = ifelse(a%%5 == 0, (a-1)%/%5+1, a%/%5+1)
    col = ifelse(a%%5 == 0, 5, a%%5)
    print(p, vp=viewport(layout.pos.row = row, layout.pos.col = col))
  }
}
dev.off()

############ 2. Neuron
genes <- intersect(Neurons_pc2_genes, Neuron_HAR_gene_targets)
pdf_file <- paste0('./PC2_gene_overlap_HAR_targets_gene_expression/Neurons_PC2_OV_NeuronHARTarget.gene_exp.pdf')  
nrow <- ceiling(length(genes) / 5)
ncol <- 5

pdf(pdf_file, height = nrow*3.5, width = ncol*3)
grid.newpage()
pushViewport(viewport(layout = grid.layout(nrow, ncol)))
a = 0
for (gene in genes){
  
  if (gene != ''){
    
    df_single_gene_exp <- df_pca_Neuron_important_gene_exp[gene, ]
    df_single_gene_exp <- reshape2::melt(df_single_gene_exp, variable.name='Sample', value.name='Exp')
    df_single_gene_exp$Species <- factor(c(rep('human', 18),
                                           rep('dog', 4),
                                           rep('mouse', 2)), levels = c('human', 'dog', 'mouse'))
    p <- plot_Neuron_boxplot(data = df_single_gene_exp, title_name = gene)
    a = a + 1
    row = ifelse(a%%5 == 0, (a-1)%/%5+1, a%/%5+1)
    col = ifelse(a%%5 == 0, 5, a%%5)
    print(p, vp=viewport(layout.pos.row = row, layout.pos.col = col))
  }
}
dev.off()
