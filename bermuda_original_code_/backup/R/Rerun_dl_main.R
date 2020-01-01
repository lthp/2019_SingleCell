
library(Seurat)
library("Matrix")
setwd("/Users/laurieprelot/Documents/Projects/2019_Deep_learning/git_DL/bermuda_original_code_/R")

source("func_read_data.R")
source("2017-08-28-runMN-US.R")

folder_name = "/Users/laurieprelot/Documents/Projects/2019_Deep_learning/git_DL/bermuda_original_code_/pancreas"
dataset_names = c("muraro_human", "baron_human")

dataset_list = list()
var_genes = list() #  a subset of highly variable genes
num_cells = 0



dataset_names = c("muraro", "baron")
for (i in c(1,2)){
seurat_csv = paste(folder_name, paste0(dataset_names[i], "_seurat.csv"), sep="/")
data = read.csv(seurat_csv, nrows = 100, header = FALSE)
data = data[4:nrow(data),]
rownames(data) = data[,1]
data = data[,2:ncol(data)]
seurat_csv = paste(folder_name, paste0(dataset_names[i], "_human_resave.csv"), sep="/")
write.csv(data, seurat_csv, sep = ",", quote = F, col.names = F, row.names = T)
}
dataset_names = c("muraro_human", "baron_human")
