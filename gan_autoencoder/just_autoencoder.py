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
from helpers_vizualisation import plot_tsne
import os

class AE():
    def __init__(self, n_markers=30):
        self.data_size = n_markers
        optimizer = Adam(0.0002, 0.5)

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mean_absolute_error',
            optimizer=optimizer)

        
    def build_generator(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.data_size))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(40))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(50))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(40))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(30))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(self.data_size, activation='tanh'))
        # model.add(Reshape(self.img_shape))
        model.summary()

        x1 = Input(shape=(self.data_size,))
        x1_gen = model(x1)

        return Model(x1, x1_gen)

    
    def train(self, x1_train_df, x1_test_df, epochs, batch_size=128, sample_interval=50):
        x1_train = x1_train_df.values
        x1_test = x1_test_df.values

        steps_per_epoch = len(x1_train) // batch_size
        for epoch in range(epochs):
            g_loss_list = []
            for step in range(steps_per_epoch):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of x1 and x2
                idx = np.random.randint(0, x1_train.shape[0], batch_size)
                x1 = x1_train[idx]

                # Generate a batch of new images
                gen_x1 = self.generator.predict(x1)
                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.generator.train_on_batch(x1, x1)
                g_loss_list.append(g_loss)
            
            # Plot the progress
            g_training_loss = np.mean(g_loss_list)
            g_validation_loss = self.generator.test_on_batch(x1_test, x1_test)
            
            print ("%d [training mae: %.4f, validation mae: %.4f]" % (epoch, g_training_loss, g_validation_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_x2(epoch, x1_train_df)
                
    def transform_batch(self, x):
        gx = self.generator.predict(x.values)
        gx_df = pd.DataFrame(data=gx, columns=x.columns, index=x.index + '_transformed')
        return gx_df

                
    def sample_x2(self, epoch, x1):
        r, c = 5, 5
        gx = self.generator.predict(x1)
        gx = pd.DataFrame(data=gx, columns=x1.columns, index=x1.index + '_transformed')
        plot_tsne(pd.concat([x1, gx]), do_pca=True, n_plots=2, iter_=500, pca_components=20, save_as='ae_epoch'+str(epoch))
        

if __name__ == '__main__':
    path = r'C:\Users\heida\Documents\ETH\Deep Learning\2019_DL_Class\code_ADAE_\chevrier_data_pooled.parquet'
    gan = GAN()
    gan.train(path, epochs=30000, batch_size=64, sample_interval=200)