from __future__ import print_function, division

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD
import tensorflow as tf


class models_gans(tf.keras.Model):  
    ''' model_type = generator, discriminator, 'combined'''
    def __init__(self, model_type):
        super(models_gans, self).__init__(name='')
        self.model_type = model_type
        self.Lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.densetanh = tf.keras.layers.Dense(20, activation='tanh', name = 'densetanh')
        self.sigmoid = tf.keras.layers.Dense(1, activation='sigmoid', name = 'sigmoid')
        self.Dense20 = tf.keras.layers.Dense(20, name = 'D20')
        self.Dense15 = tf.keras.layers.Dense(15, name = 'D15')
        self.Dense10 = tf.keras.layers.Dense(10, name = 'D10')
        self.Dense512 = tf.keras.layers.Dense(512, name = 'D512')
        self.Dense256 = tf.keras.layers.Dense(256, name = 'D256')
        self.Dense20bis = tf.keras.layers.Dense(20, name = 'D20bis')
    
    def generator(self, input_tensor, training = True):
        x = self.Dense20(input_tensor)
        x = self.Lrelu(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        
        x = self.Dense15(x)
        x = self.Lrelu(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

        x = self.Dense20bis(x)
        x = self.Lrelu(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

        x = tf.keras.layers.Add(x, input_tensor)
        #x += input_tensor # ADD BACK 
        
        x = self.densetanh(x)
        x.trainable = training
        return x
        
    def discriminator(self, input_tensor, training = True):

        x = self.Dense512(input_tensor)
        x = self.Lrelu(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        
        x = self.Dense256(x)
        x = self.Lrelu(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        
        x = self.sigmoid(x)
        x.trainable = training
        return x
        
        
        
    def call(self, input_tensor, training = False):
        if self.model_type == 'generator':
            x = self.generator(input_tensor, training = True)
        elif self.model_type == 'discriminator':
            x = self.discriminator(input_tensor, training = True)

        if self.model_type == 'combined':
            x = self.generator(input_tensor, training = False)
            x = self.discriminator(x, training = True)
        return x
    
    

class GAN_residuals():
    ''' batch1 and batch2 are pandas dataframes with batch 1 and batch 2 respectively and no metadata or index'''
    def __init__(self, batch1, batch2):
        # batches
        self.batch1 = batch1
        self.batch2 = batch2
        
        #parameters 
        self.data_size = batch1.shape[1]
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = models_gans('discriminator')
        _ = self.discriminator(tf.zeros((1, batch1.shape[1])))
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        # Build the generator
        self.generator = models_gans('generator')
        x1 = Input(shape=(self.data_size,))
        gen_x1 = self.generator(x1)


        # The generator takes x1 as input and generates gen_x1 (fake x2)
        self.combined = models_gans('combined')

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        _ = self.combined(tf.zeros((1, batch1.shape[1])))
        self.combined.compile(loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

        
    def train(self, epochs, batch_size=128, sample_interval=50):
        plot_model = {"epoch":[],"d_loss":[],"g_loss":[]}
        x1 = self.batch1.values
        x2 = self.batch2.values

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of x1 and x2
            idx = np.random.randint(0, x1.shape[0], batch_size)
            x1 = x1[idx]
            x2 = x2[idx]

            # Generate a batch of new images
            gen_x1 = self.generator.predict(x1, verbose = 1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(x2, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_x1, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(x1, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
            plot_model["epoch"].append(epoch)
            plot_model["d_loss"].append(d_loss[0])
            plot_model["g_loss"].append(g_loss[0])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_x2(epoch, x1)
        return plot_model

    def sample_x2(self, epoch, x1):
        r, c = 5, 5
        gen_x2 = self.generator.predict(x1)
        # can e.g. save some pca figures of the gen_x2


if __name__ == '__main__':
    import os
    from loading_and_preprocessing.data_loader import load_data_basic, load_data_cytof
    # path = r'C:\Users\heida\Documents\ETH\Deep Learning\2019_DL_Class_old\code_ADAE_\chevrier_data_pooled_panels.parquet'
    path = r'C:\Users\heida\Documents\ETH\Deep Learning\2019_DL_Class_old\code_ADAE_\chevrier_data_pooled_panels.parquet'
    x1_train, x1_test, x2_train, x2_test = load_data_cytof(path, patient_id='rcc7', n=10000)

    #path = os.getcwd()
    #path = path + '/toy_data_gamma_small.parquet'  # '/toy_data_gamma_large.parquet'
    #x1_train, x1_test, x2_train, x2_test = load_data_basic(path, patient='sample1', batch_names=['batch1', 'batch2'],
    #                                                       seed=42, n_cells_to_select=0)
    gan = GAN_residuals(x1_train, x2_train)
    gan.train(epochs=3000, batch_size=64, sample_interval=50)

