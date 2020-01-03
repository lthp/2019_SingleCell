#!/bin/bash
source activate deeplearning
data_loc = '/cluster/home/hthrainsson/chevrier_data_pooled_full_panels.parquet'

lambda = 0.9
sample_no = 5
echo "python residual_gan/residual_gan_autoencoder.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
echo "python residual_gan/residual_gan_autoencoder_without_residuals.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"

sample_no = 65
echo "python residual_gan/residual_gan_autoencoder.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
echo "python residual_gan/residual_gan_autoencoder_without_residuals.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"

sample_no = 75
echo "python residual_gan/residual_gan_autoencoder.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
echo "python residual_gan/residual_gan_autoencoder_without_residuals.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"

lambda = 0.8
sample_no = 5
echo "python residual_gan/residual_gan_autoencoder.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
echo "python residual_gan/residual_gan_autoencoder_without_residuals.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"

sample_no = 65
echo "python residual_gan/residual_gan_autoencoder.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
echo "python residual_gan/residual_gan_autoencoder_without_residuals.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"

sample_no = 75
echo "python residual_gan/residual_gan_autoencoder.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
echo "python residual_gan/residual_gan_autoencoder_without_residuals.py ${sample_no} ${data_loc} ${lambda}"|  bsub -n 1 -W 24:00 -R "rusage[mem=5000]"
