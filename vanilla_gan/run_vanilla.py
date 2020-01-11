#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from vanilla_gan.gan_batches_optimized import GAN
from loading_and_preprocessing.data_loader import load_data_basic
path = 'data\chevrier_samples_5_65_75.parquet'
sample_names = ['sample5', 'sample65', 'sample75']
batch_names = ['batch1', 'batch3']

for sample_name in sample_names:
    x1_train, x1_test, x2_train, x2_test = load_data_basic(path, sample=sample_name,
                                                           batch_names=batch_names, seed=42, panel=None,
                                                           upsample=True)
    gan = GAN(x1_train.shape[1], modelname='gan_vanilla_full')
    gan.train(x1_train, x2_train, epochs=1000, batch_size=64, sample_interval=50)