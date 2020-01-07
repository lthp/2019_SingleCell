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

3) Residual GANs: To run the residual gan with skip connections 
```
python residual_gan/run_residuals.py
```
To run the residual gan without skip connections
```
python residual_gan/run_residuals_wo_res.py
```
To run both versions on the cluster
```
residual_gan_chevrier_cluster.sh
```

4) Evaluation: Move the score files of the models from their respective "output_<modelname>" files and 
into "eval_scores". Run the "notebooks/find_best_epochs.ipynb" notebook to get a table of the scores and 
see the best performing epoch for each model. For each model, copy the gx1 of the best epoch along with gx1,
x1 and x2 of epoch 0 to the "final_plots/chevrier/dataframes_for_plotting" folder. To generate the tsne plots 
colored by batch and cell type, run "python visualisation_and_evaluation/generate_plots_chevrier.py"



All of the runs will output for every model on every 50th epoch, 
the divergence score, entropy score, and silhoutte score, 
as well as the raw data-frames, values of the losses, and tsne and umap plots.
