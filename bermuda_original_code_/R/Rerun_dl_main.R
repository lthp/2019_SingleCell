
library(Seurat)
library("Matrix")
setwd("/Users/laurieprelot/Documents/Projects/2019_Deep_learning/git_DL/bermuda_original_code_/R")

source("func_read_data.R")
source("2017-08-28-runMN-US.R")

folder_name = "/Users/laurieprelot/Documents/Projects/2019_Deep_learning/git_DL/bermuda_original_code_/pancreas"


dataset_list = list()
var_genes = list() #  a subset of highly variable genes
num_cells = 0


### Resave the dataframes
dataset_names = c("muraro", "baron")
for (i in c(1,2)){
  seurat_csv = paste(folder_name, paste0(dataset_names[i], "_seurat.csv"), sep="/")
  data = read.table(seurat_csv, sep = ',')#, col.names = F)
  data = data[c(1,2, 4:nrow(data)),]
  rownames(data) = data[,1]
  data = data[,2:ncol(data)]
  seurat_csv = paste(folder_name, paste0(dataset_names[i], "_human_resave.csv"), sep="/")
  write.table(data, seurat_csv, sep = ",", quote = F, col.names = F, row.names = T)
}


### Cluster list extract 
cluster_list = list()
dataset_names = c("muraro", "baron")
for (i in c(1,2)){
  seurat_csv = paste(folder_name, paste0(dataset_names[i], "_seurat.csv"), sep="/")
  data = read.table(seurat_csv, sep = ',')#, col.names = F)
  cluster_list[[i]] = as.integer(data[3,2:dim(data)[2]])
}
cluster_label_list = cluster_list
  