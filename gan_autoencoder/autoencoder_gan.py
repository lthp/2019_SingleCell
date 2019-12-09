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


class GAN():
    def __init__(self, n_markers=30):
        self.data_size = n_markers
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes x1 as input and generates gen_x1 (fake x2)
        x1 = Input(shape=(self.data_size,))
        gen_x1 = self.generator(x1)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated data as input and determines validity
        validity = self.discriminator(gen_x1)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(inputs=x1, outputs=[gen_x1, validity])
        losses = ['mean_absolute_error', 'binary_crossentropy']
        loss_weights = [1, 0.1]
        self.combined.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights)

        
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

    
    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.data_size,)))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        x2 = Input(shape=self.data_size)
        validity = model(x2)
        return Model(x2, validity)

    
    def train(self, x1, x2, epochs, batch_size=128, sample_interval=50):
        x1_train = x1.values
        x2_train = x2.values
        plot_model = {"epoch":[],"d_loss":[],"g_loss":[]}

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        d_loss = [0, 0]
        
        steps_per_epoch = len(x1_train) // batch_size
        for epoch in range(epochs):

            for step in range(steps_per_epoch):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of x1 and x2
                idx = np.random.randint(0, x1_train.shape[0], batch_size)
                x1 = x1_train[idx]
                x2 = x2_train[idx]

                # Generate a batch of new images
                gen_x1 = self.generator.predict(x1)

                # Train the discriminator
                if d_loss[1] > 0.8:
                    d_loss_real = self.discriminator.test_on_batch(x2, valid)
                    d_loss_fake = self.discriminator.test_on_batch(gen_x1, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    pass
                else:
                    d_loss_real = self.discriminator.train_on_batch(x2, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_x1, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(x1, [x1, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, mae: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[0]))
            plot_model["epoch"].append(epoch)
            plot_model["d_loss"].append(d_loss[0])
            plot_model["g_loss"].append(g_loss[1])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_x2(epoch, x1)
        return plot_model

                
    def transform_batch(self, x):
        gx = self.generator.predict(x)
        gx_df = pd.DataFrame(data=gx, columns=x.columns, index=x.index + '_transformed')
        return gx_df

                
    def sample_x2(self, epoch, x1):
        r, c = 5, 5
        gen_x2 = self.generator.predict(x1)
        # can e.g. save some pca figures of the gen_x2


if __name__ == '__main__':
    path = r'C:\Users\heida\Documents\ETH\Deep Learning\2019_DL_Class\code_ADAE_\chevrier_data_pooled.parquet'
    gan = GAN()
    gan.train(path, epochs=30000, batch_size=64, sample_interval=200)