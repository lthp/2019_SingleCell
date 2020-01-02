from __future__ import print_function, division

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from info_on_checkpoint import save_info, save_plots
from visualisation_and_evaluation.helpers_vizualisation import plot_tsne, plot_metrics, plot_umap
import numpy as np
import pandas as pd
from datetime import datetime

'''
This model has the exact same structure and optimizations as "autoencoder_gan_reconstructionloss", 
but the only difference is the loss - just keep the vanilla loss, not have reconstruction loss.
'''

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
        self.combined = Model(inputs=x1, outputs=validity)
        # g_loss = self.combined.train_on_batch(x1, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, )

    def build_generator(self):
        model = Sequential()
        # model.add(Dense(30, input_dim=self.data_size))
        model.add(Dense(35, input_dim=self.data_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # model.add(Dense(40))
        model.add(Dense(45))
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Dense(50))
        model.add(Dense(55))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # model.add(Dense(40))
        model.add(Dense(45))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # model.add(Dense(30))
        model.add(Dense(35))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.data_size, activation='tanh'))
        # model.add(Reshape(self.img_shape))
        model.summary()

        x1 = Input(shape=(self.data_size,))
        x1_gen = model(x1)

        return Model(x1, x1_gen, name='generator')

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

    def train(self, x1_df, x2_df, epochs, batch_size=128, sample_interval=50):
        fname = datetime.now().strftime("%d-%m-%Y_%H.%M.%S")
        #fname = '_ganvanilladiamondbatch_full_corrupsample' + x1_df.index[0].split('.')[0]
        #fname = time + fname
        os.makedirs(os.path.join('figures', fname))
        os.makedirs(os.path.join('output', fname))
        os.makedirs(os.path.join('models', fname))
        plot_model = {"epoch": [], "d_loss": [], "g_loss": []}
        training_metrics = {"epoch": [], "d_loss": [], "d_accuracy": [], "g_loss": []}

        x1_train = x1_df.values
        x2_train = x2_df.values

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
                g_loss = self.combined.train_on_batch(x1, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            training_metrics["epoch"].append(epoch)
            training_metrics["d_loss"].append(d_loss[0])
            training_metrics["d_accuracy"].append(d_loss[1])
            training_metrics["g_loss"].append(g_loss)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print('generating plots and saving outputs')
                gx1 = self.generator.predict(x1_df)
                save_info.save_dataframes(epoch, x1_df, x2_df, gx1, fname, dir_name='output')
                save_info.save_dataframes(epoch, x1_df, x2_df, gx1, fname, dir_name='output')
                save_info.save_scores(epoch, x1_df, x2_df, gx1, training_metrics, fname, dir_name='output')


if __name__ == '__main__':
    import os
    from loading_and_preprocessing.data_loader import load_data_basic, load_data_cytof
    # path = r'C:\Users\heida\Documents\ETH\Deep Learning\2019_DL_Class_old\code_ADAE_\chevrier_data_pooled_panels.parquet'
    path = r'C:\Users\Public\PycharmProjects\deep\Legacy_2019_DL_Class\data\chevrier_data_pooled_full_panels.parquet'
    x1_train, x1_test, x2_train, x2_test = load_data_basic(path, sample='sample75', batch_names=['batch1', 'batch3'],
                                                           seed=42, panel=None)

    #path = os.getcwd()
    #path = path + '/toy_data_gamma_small.parquet'  # '/toy_data_gamma_large.parquet'
    #x1_train, x1_test, x2_train, x2_test = load_data_basic(path, patient='sample1', batch_names=['batch1', 'batch2'],
    #                                                       seed=42, n_cells_to_select=0)
    gan = GAN(x1_train.shape[1])
    gan.train(x1_train, x2_train, epochs=3000, batch_size=64, sample_interval=50)


