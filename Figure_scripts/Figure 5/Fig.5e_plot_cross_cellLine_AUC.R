setwd('/data/xujs/Project/HiC2PoreC/code/H2P/Analysis_Results/2024-07-02_cross_cellLine_prediction/')

library(ggplot2)
library(ComplexHeatmap)
library(pheatmap)

df_cross_cellLine_1Mb <-  data.frame(GM12878 = c(0.8100, 0.7736, 0.7695, 0.7647),
                                     K562 =    c(0.7943, 0.8053, 0.7796, 0.7769),
                                     hESC =    c(0.7431, 0.7203, 0.8074, 0.7973),
                                     mESC  = c(0.7305, 0.7149, 0.7872, 0.8396),
                                     row.names = c('GM12878', 'K562', 'hESC', 'mESC'))

df_cross_cellLine_100kb <-  data.frame(GM12878 = c(0.9218, 0.8955, 0.8351, 0.8363),
                                       K562 =    c(0.9170, 0.9212, 0.8609, 0.8423),
                                       hESC =    c(0.8918, 0.8657, 0.9068, 0.8650),
                                       mESC  =   c(0.8068, 0.7962, 0.8193, 0.8576),
                                       row.names = c('GM12878', 'K562', 'hESC', 'mESC'))

df_cross_cellLine_5kb <-  data.frame(GM12878 = c(0.9423, 0.8922, 0.8591, 0.8529),
                                     K562 =    c(0.9045, 0.9417, 0.8606, 0.8536),
                                     hESC =    c(0.8924, 0.8713, 0.9177, 0.8495),
                                     mESC  =   c(0.8135, 0.8248, 0.8353, 0.8691),
                                     row.names = c('GM12878', 'K562', 'hESC', 'mESC'))

### 1Mb
pdf('H2P_Cross_cellLine_prediction_1Mb.heatmap.pdf', height = 5, width=6)
p_1Mb <- Heatmap(df_cross_cellLine_1Mb,
                 border = NA, 
                 #column_split = data.frame(time=rep(c(1,2,3),c(9,2,30))),
                 #column_gap = unit(c(3,3), 'mm'), 
                 #column_title = NULL, row_title = c("Cluster1", "Cluster2", "Cluster3"),
                 #row_gap = unit(c(3, 3), 'mm'),
                 name = 'H2P cross_cellLine',
                 cluster_rows = F, cluster_columns = F,
                 show_row_names = T, row_names_gp = gpar(fontsize = 12),
                 show_column_names = T, column_names_gp = gpar(fontsize = 12),
                 # clustering_distance_rows = 'spearman', clustering_method_rows = 'complete',
                 # row_dend_reorder = T,
                 col = colorRamp2(seq(0.7, 0.85, length.out=100),
                                  colorRampPalette(c('#2d76bc', 'white', '#c41f1a'))(100)),
                 cell_fun = function(i, j, x, y, width, height, fill){
                   grid.text(sprintf("%.4f", df_cross_cellLine_1Mb[j, i]), x, y, gp=gpar(fontsize=14))
                 }
)
draw(p_1Mb, column_title='H2P Cross cellLine prediction (1Mb)', column_title_gp=gpar(fontsize=14))
dev.off()

# col = colorRamp2(seq(0.8, 0.95, length.out=100),
#                  colorRampPalette(c('dodgerblue', 'black', 'yellow'))(100)),

### 100kb
pdf('H2P_Cross_cellLine_prediction_100kb.heatmap.pdf', height = 5, width=6)
p_100kb <- Heatmap(df_cross_cellLine_100kb,
                 border = NA, 
                 #column_split = data.frame(time=rep(c(1,2,3),c(9,2,30))),
                 #column_gap = unit(c(3,3), 'mm'), 
                 #column_title = NULL, row_title = c("Cluster1", "Cluster2", "Cluster3"),
                 #row_gap = unit(c(3, 3), 'mm'),
                 name = 'H2P cross_cellLine',
                 cluster_rows = F, cluster_columns = F,
                 show_row_names = T, row_names_gp = gpar(fontsize = 12),
                 show_column_names = T, column_names_gp = gpar(fontsize = 12),
                 # clustering_distance_rows = 'spearman', clustering_method_rows = 'complete',
                 # row_dend_reorder = T,
                 col = colorRamp2(seq(0.75, 0.95, length.out=100),
                                  colorRampPalette(c('#2d76bc', 'white', '#c41f1a'))(100)),
                 cell_fun = function(i, j, x, y, width, height, fill){
                   grid.text(sprintf("%.4f", df_cross_cellLine_100kb[j, i]), x, y, gp=gpar(fontsize=14))
                 }
)
draw(p_100kb, column_title='H2P Cross cellLine prediction (100kb)', column_title_gp=gpar(fontsize=14))
dev.off()


### 5kb
pdf('H2P_Cross_cellLine_prediction_5kb.heatmap.pdf', height = 5, width=6)
p_5kb <- Heatmap(df_cross_cellLine_5kb,
                 border = NA, 
                 #column_split = data.frame(time=rep(c(1,2,3),c(9,2,30))),
                 #column_gap = unit(c(3,3), 'mm'), 
                 #column_title = NULL, row_title = c("Cluster1", "Cluster2", "Cluster3"),
                 #row_gap = unit(c(3, 3), 'mm'),
                 name = 'H2P cross_cellLine',
                 cluster_rows = F, cluster_columns = F,
                 show_row_names = T, row_names_gp = gpar(fontsize = 12),
                 show_column_names = T, column_names_gp = gpar(fontsize = 12),
                 # clustering_distance_rows = 'spearman', clustering_method_rows = 'complete',
                 # row_dend_reorder = T,
                col = colorRamp2(seq(0.8, 0.95, length.out=100),
                                 colorRampPalette(c('#2d76bc', 'white', '#c41f1a'))(100)),
                cell_fun = function(i, j, x, y, width, height, fill){
                  grid.text(sprintf("%.4f", df_cross_cellLine_5kb[j, i]), x, y, gp=gpar(fontsize=14))
                }
)
draw(p_5kb, column_title='H2P Cross cellLine prediction (5kb)', column_title_gp=gpar(fontsize=14))
dev.off()
# col = colorRamp2(seq(0.75, 1.0, length.out=100), colorRampPalette(c('dodgerblue', 'black', 'yellow'))(100)),
