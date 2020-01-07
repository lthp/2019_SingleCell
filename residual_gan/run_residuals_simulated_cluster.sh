#!/bin/bash
source activate deeplearning
data_loc='/cluster/home/hthrainsson/toy_data_gamma_w_index.parquet'
sample_no=1
lambda=0.9

echo "python residual_gan/residual_gan_autoencoder.py ${sample_no} ${data_loc} ${lambda} --toy"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
echo "python residual_gan/residual_gan_autoencoder_without_residuals.py ${sample_no} ${data_loc} ${lambda} --toy"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"

lambda=0.8
echo "python residual_gan/residual_gan_autoencoder.py ${sample_no} ${data_loc} ${lambda} --toy"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
echo "python residual_gan/residual_gan_autoencoder_without_residuals.py ${sample_no} ${data_loc} ${lambda} --toy"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"

