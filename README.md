# Course Project in the 2019 Deep Learning Course, ETH.
Our project is to impelement GANs for batch effect correction in CyTOF data.
The project consists of a 5-page report with a supplementary, 
as well as code, that can be found at the github: https://github.com/lthp/2019_DL_Class.
The datasets used are also on the github, under the folder data.

To run the models described in the paper 
(2x autoencoder-gans, vanilla-gan, 2x residual-gans, 1x bermuda-inspired gan), 
run the file XXX.

This run will output for every model on every 50th epoch, the divergence score, entropy score, and silhoutte score, 
as well as the raw data-frames, values of the losses, and tsne and umap plots.

To calculate combined scores of best epoch for comparison of models, 
run the file evaluation.ipynb (under the notebooks directory). 


