#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from residual_gan.residual_gan_autoencoder_without_residuals import GAN
from loading_and_preprocessing.data_loader import load_data_basic
loss_lambda = 0.8
path = '..\data\chevrier_samples_5_65_75.parquet'
sample_names = ['sample5', 'sample65', 'sample75']
batch_names = ['batch1', 'batch3']
modelname = 'residual_gan_wo_res_full'
for sample_name in sample_names:
    x1_train, x1_test, x2_train, x2_test = load_data_basic(path, sample=sample_name,
                                                           batch_names=batch_names, seed=42, panel=None,
                                                           upsample=True)
    gan = GAN(modelname, x1_train.shape[1], loss_lambda)
    gan.train(x1_train, x2_train, epochs=1000, batch_size=64, sample_interval=50)
