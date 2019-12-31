

folder_name = '/Users/laurieprelot/Documents/Projects/2019_Deep_learning/data/Chevrier-et-al'
file = 'chevrier_data_pooled_full_panels.batch1_batch3.bermuda.tsv'
filename = paste(folder_name, file, sep="/")
print(paste0("Dataset: ", filename))
# Seurat
dataset = read.table(filename, sep = '\t', nrows = 10, header = F)
rownames(dataset) = dataset[,1]
dataset = dataset[,2:ncol(dataset)]


# Metaneighbor
cluster_labels = unique(as.integer(dataset["metadata_phenograph",])) # Unique 1-21 

pheno = as.data.frame(list(Celltype =  as.character(dataset["metadata_phenograph",]),# cluster id 
                           Study_ID =  as.character(dataset["dataset_label",])), #dataset label
                      stringsAsFactors=FALSE) # Length = all the cells of dataset 1 + dataset 2


wrap_MetaNeighbor<-function(var_genes, data, cluster_labels, pheno){
  # run metaneighbor
  
  #var_genes = rep(var_genes[1], 100)
  cluster_similarity = run_MetaNeighbor_US(var_genes, data, cluster_labels, pheno)
  
  # set cluster pairs from the same dataset to 0
  for (i in 1:length(dataset_list)) {
    cluster_idx_tmp = unique(cluster_label_list[[i]])
    cluster_similarity[cluster_idx_tmp, cluster_idx_tmp] = 0
  }
  
  # order rows and columns
  cluster_similarity = cluster_similarity[order(as.numeric(rownames(cluster_similarity))),]
  cluster_similarity = cluster_similarity[,order(as.numeric(colnames(cluster_similarity)))]
  
  # write out metaneighbor file
  metaneighbor_file = paste(folder_name, paste0(folder_name, "_metaneighbor.csv"), sep="/")
  write.table(cluster_similarity, metaneighbor_file, sep = ",", quote = F, col.names = T, row.names = F)
}





# Code from
# https://github.com/maggiecrow/MetaNeighbor

run_MetaNeighbor_US<-function(vargenes, data, celltypes, pheno){
  
  cell.labels=matrix(0,ncol=length(celltypes),nrow=dim(pheno)[1])
  rownames(cell.labels)=colnames(data)
  colnames(cell.labels)=celltypes
  for(i in 1:length(celltypes)){
    type=celltypes[i]
    m<-match(pheno$Celltype,type)
    cell.labels[!is.na(m),i]=1
  }
  
  m<-match(rownames(data),vargenes)
  cor.dat=cor(data[!is.na(m),],method="s")
  rank.dat=cor.dat*0
  rank.dat[]=rank(cor.dat,ties.method="average",na.last = "keep")
  rank.dat[is.na(rank.dat)]=0
  rank.dat=rank.dat/max(rank.dat)
  sumin    =  (rank.dat) %*% cell.labels
  sumall   = matrix(apply(rank.dat,2,sum), ncol = dim(sumin)[2], nrow=dim(sumin)[1])
  predicts = sumin/sumall
  
  cell.NV=matrix(0,ncol=length(celltypes),nrow=length(celltypes))
  colnames(cell.NV)=colnames(cell.labels)
  rownames(cell.NV)=colnames(cell.labels)
  
  for(i in 1:dim(cell.labels)[2]){
    predicts.temp=predicts
    m<-match(pheno$Celltype,colnames(cell.labels)[i])
    study=unique(pheno[!is.na(m),"Study_ID"])
    m<-match(pheno$Study_ID,study)
    pheno2=pheno[!is.na(m),]
    predicts.temp=predicts.temp[!is.na(m),]
    predicts.temp=apply(abs(predicts.temp), 2, rank,na.last="keep",ties.method="average")
    filter=matrix(0,ncol=length(celltypes),nrow=dim(pheno2)[1])
    m<-match(pheno2$Celltype,colnames(cell.labels)[i])
    filter[!is.na(m),1:length(celltypes)]=1
    negatives = which(filter == 0, arr.ind=T)
    positives = which(filter == 1, arr.ind=T)
    predicts.temp[negatives] <- 0
    np = colSums(filter,na.rm=T)
    nn = apply(filter,2,function(x) sum(x==0,na.rm=T))
    p =  apply(predicts.temp,2,sum,na.rm=T)
    cell.NV[i,]= (p/np - (np+1)/2)/nn
  }
  
  cell.NV=(cell.NV+t(cell.NV))/2
  return(cell.NV)
  
}
