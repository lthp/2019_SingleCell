#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from gan_autoencoder.diamond_batchnorm import GAN as GAN_wider
from gan_autoencoder.autoencoder_gan_reconstructionloss_bottleneck import GAN as GAN_norrower
from loading_and_preprocessing.data_loader import load_data_basic
path = 'data\chevrier_samples_5_65_75.parquet'
sample_names = ['sample5', 'sample65', 'sample75']
batch_names = ['batch1', 'batch3']

# train GAN autoencoder diamond shape wider

print('Train GAN autoencoder diamond shape wider')
for sample_name in sample_names:
    x1_train, x1_test, x2_train, x2_test = load_data_basic(path, sample=sample_name,
                                                           batch_names=batch_names, seed=42, panel=None,
                                                           upsample=True)
    gan = GAN_wider(x1_train.shape[1], modelname='gan_autoencoder_diamond_wider_full')
    gan.train(x1_train, x2_train, epochs=1000, batch_size=64, sample_interval=50)


print('Train GAN autoencoder diamond shape narrower')
for sample_name in sample_names:
    x1_train, x1_test, x2_train, x2_test = load_data_basic(path, sample=sample_name,
                                                           batch_names=batch_names, seed=42, panel=None,
                                                           upsample=True)
    gan = GAN_norrower(x1_train.shape[1], modelname='gan_autoencoder_diamond_narrower_full')
    gan.train(x1_train, x2_train, epochs=1000, batch_size=64, sample_interval=50)