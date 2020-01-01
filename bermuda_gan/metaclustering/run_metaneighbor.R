#folder_name = '/Users/laurieprelot/Documents/Projects/2019_Deep_learning/data/Chevrier-et-al'
folder_name='/cluster/work/grlab/projects/tmp_laurie/dl_data' 

file = 'chevrier_data_pooled_full_panels.batch1_batch3.bermuda.tsv'



# Code from
# https://github.com/maggiecrow/MetaNeighbor

run_MetaNeighbor_US<-function(vargenes, data, celltypes, pheno){
  
  cell.labels=matrix(0,ncol=length(celltypes),nrow=dim(pheno)[1])
  rownames(cell.labels)=colnames(data)
  colnames(cell.labels)=celltypes
  print("...prepare match")
  for(i in 1:length(celltypes)){
    type=celltypes[i]
    m<-match(pheno$Celltype,type)
    cell.labels[!is.na(m),i]=1 # cell * clusters with hot encoding 
  }
  m<-match(rownames(data),vargenes)
  print("...run correlation")
  cor.dat=cor(data[!is.na(m),],method="s")
  rank.dat=cor.dat*0
  print("...run rank")
  rank.dat[]=rank(cor.dat,ties.method="average",na.last = "keep") # cell* cell correlation item 
  print(rank.dat[1:10, 1:10])
  rank.dat[is.na(rank.dat)]=0
  rank.dat=rank.dat/max(rank.dat)
  sumin    =  (rank.dat) %*% cell.labels
  print('...run sum ranks')
  sumall   = matrix(apply(rank.dat,2,sum), ncol = dim(sumin)[2], nrow=dim(sumin)[1])
  predicts = sumin/sumall
  
  cell.NV=matrix(0,ncol=length(celltypes),nrow=length(celltypes))
  colnames(cell.NV)=colnames(cell.labels)
  rownames(cell.NV)=colnames(cell.labels)
  
  print("... run predict")
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
  print("...metaneighbor done")
  return(cell.NV)
  
}

preprocessing<-function(folder_name, file){

filename = paste(folder_name, file, sep="/")
print(paste0("Dataset: ", filename))

# preprocessing
dataset = read.table(filename, sep = '\t', header = F)
rownames(dataset) = dataset[,1]
dataset = dataset[,2:ncol(dataset)]
print("subsetting dataset")
dataset = dataset[, c(1:15000, 33000:48000)]#TODO remove
cluster_labels = unique(as.integer(dataset["metadata_phenograph",])) # Unique 1-21 

pheno = as.data.frame(list(Celltype =  as.character(dataset["metadata_phenograph",]),# cluster id 
                           Study_ID =  as.character(dataset["dataset_label",])), #dataset label
                      stringsAsFactors=FALSE) # Length = all the cells of dataset 1 + dataset 2
data = dataset[5:nrow(dataset), ]
var_genes = rownames(data) 
print("... data preprocessed")
return(list(var_genes=var_genes, data=data, cluster_labels=cluster_labels, pheno=pheno))
}



####___________Main ___________####
wrap_MetaNeighbor<-function(folder_name, file){
  ### Preprocessing
  inputs = preprocessing(folder_name, file)
  var_genes=inputs$var_genes
  data=inputs$data
  cluster_labels=inputs$cluster_labels
  pheno=inputs$pheno
  write.table(pheno, paste(folder_name, "pheno.tsv", sep="/"), sep = "\t", quote = F, col.names = T, row.names = F)
  
  ### run metaneighbor
  cluster_similarity = run_MetaNeighbor_US(var_genes, data, cluster_labels, pheno)
  print(cluster_similarity[1:10, 1:10])
  
  ### set cluster pairs from the same dataset to 0
  for (i in 1:dim(unique(pheno["Study_ID"]))[1] ) {
    cluster_idx_tmp = as.integer(unique(pheno[pheno["Study_ID"] == i, "Celltype"])) #unique(cluster_label_list[[i]])
    pos = match(cluster_idx_tmp, cluster_labels)
    print(cluster_idx_tmp)
    print(pos)
    cluster_similarity[pos, pos] = 0
  }
  
  ### order rows and columns
  cluster_similarity = cluster_similarity[order(as.numeric(rownames(cluster_similarity))),]
  cluster_similarity = cluster_similarity[,order(as.numeric(colnames(cluster_similarity)))]
  
  ### write out metaneighbor file
  print("writing outputs to:")
  splitted = strsplit(c(file), ".", fixed = T)
  splitted = splitted[[1]]
  base_name = paste( splitted[1:length(splitted) -1], collapse = '.')
  metaneighbor_file = paste(folder_name, paste0(base_name, "_metaneighbor.tsv"), sep="/")
  print(metaneighbor_file)
  write.table(cluster_similarity, metaneighbor_file, sep = "\t", quote = F, col.names = T, row.names = F)
}



### Run 
wrap_MetaNeighbor(folder_name, file)
