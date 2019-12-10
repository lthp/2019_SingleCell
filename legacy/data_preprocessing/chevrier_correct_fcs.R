# script to correct corrupted .fcs files to be able to open them in python
library('flowCore')
path='/Users/joannaf/Desktop/DeepLearning/DL2019/project/data/Dataset5/data_raw/'
path_save = '/Users/joannaf/Desktop/DeepLearning/DL2019/project/data/Dataset5/data_corrected/'
files = list.files(path, recursive=TRUE, pattern='.fcs')
for (f in files){
  fname = paste0(path, f)
  df = read.FCS(fname, transformation=FALSE)
  fname_save = paste0(path_save, f)
  print(fname_save)
  write.FCS(df, fname_save)
}
