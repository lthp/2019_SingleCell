import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Concatenate
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU
from tensorflow.python.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python.keras import metrics
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam

import pdb
import os

from gan_autoencoder.helpers_vizualisation import eval_knn_proportions
from gan_autoencoder.helpers_vizualisation import plot_tsne
from gan_autoencoder.helpers_vizualisation import plot_umap

from gan_autoencoder.data_loader import load_data_basic, normalize

from gan_autoencoder.autoencoder_gan_reconstructionloss import GAN

path = os.getcwd()
path = path+'/toy_data_gamma_small.parquet' # '/toy_data_gamma_large.parquet'
x1_train, x1_test, x2_train, x2_test = load_data_basic(path, patient='sample1', batch_names = ['batch1', 'batch2'], seed=42,
                                                      n_cells_to_select=0)
gan = GAN(x1_train.shape[1])
#gan.train(x1, x2, epochs=30000, batch_size=64, sample_interval=200)
gan.train(x1_train, x2_train, epochs=3000, batch_size=64, sample_interval=200)
x1_train_transformed = gan.transform_batch(x1_train)
f = 5
plt.figure()
plt.hist(x2_train.values[:, f])
plt.figure()
plt.hist(x1_train.values[:, f])
plt.figure()
plt.hist(x1_train_transformed.values[:, f])
x_train = pd.concat([x1_train, x1_train_transformed])
plot_tsne(x_train, do_pca=True, n_plots=2, iter_=500, pca_components=20)

x_train = pd.concat([x1_train, x2_train])
plot_tsne(x_train, do_pca=True, n_plots=2, iter_=500, pca_components=20)

x_train = pd.concat([x1_train_transformed, x2_train])
plot_tsne(x_train, do_pca=True, n_plots=2, iter_=500, pca_components=20)

plot_umap(pd.concat([x1_train, x2_train]))
plot_umap(pd.concat([x1_train_transformed, x2_train]))