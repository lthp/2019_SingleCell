# Course Project in the 2019 Deep Learning Course, ETH.
Our project is to impelement GANs for batch effect correction in CyTOF data.
The project consists of a 5-page report with a supplementary, 
as well as code, that can be found at the github: https://github.com/lthp/2019_DL_Class.
A link to the datasets used are provided in the hand-in.
Place the dataset in a subfolder "/data", so everything will run smoothly.

To run the models described in the paper:
1) Autoencoder GANs: 
Run the wider diamond shape model on all the 3 samples of the real-world data via
                 python gan_autoencoder/diamond_batchnorm.py
Run the flatter diamond shape model on all the 3 samples of the real-world data via               
                 python gan_autoencoder/autoencoder_gan_reconstructionloss_bottleneck.py

2) Vanilla GAN: Run the vanilla gan model on all the 3 sample of the real world data via
                python vanilla_gan/gan_batches_optimized.py

3) Residual GANs:

All of the runs will output for every model on every 50th epoch, 
the divergence score, entropy score, and silhoutte score, 
as well as the raw data-frames, values of the losses, and tsne and umap plots.

To calculate combined scores of best epoch for comparison of models, 
run the file evaluation.ipynb (under the notebooks directory). 



