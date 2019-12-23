# !/usr/bin/env python
from __future__ import print_function, division

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from visualisation_and_evaluation.helpers_vizualisation import plot_tsne, plot_metrics, plot_umap
import numpy as np
import pandas as pd
from datetime import datetime
from info_on_checkpoint import save_info, save_plots


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

    def residual_block(self, number_of_nodes):
        x1 = Input(shape=(self.data_size,))
        x = x1
        for n in number_of_nodes:
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dense(n)(x)

        #x = Dense(self.data_size)(x)
        x = Activation('tanh')(x)

        x = Add()([x, x1])
        return Model(x1, x)


    def build_generator(self):
        x1 = Input(shape=(self.data_size,))

        block1 = self.residual_block([40, self.data_size])(x1)
        block2 = self.residual_block([40, self.data_size])(block1)

        # model.add(Reshape(self.img_shape))

        model = Model(x1, block2)
        model.summary()

        x1_gen = model(x1)

        return Model(x1, x1_gen)


    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.data_size,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        x2 = Input(shape=self.data_size)
        validity = model(x2)
        return Model(x2, validity)


    def train(self, x1_train_df, x2_train_df, epochs, batch_size=128, sample_interval=50):
        fname = datetime.now().strftime("%d-%m-%Y_%H.%M.%S")
        os.makedirs(os.path.join('figures_resgan', fname))
        os.makedirs(os.path.join('output_resgan', fname))
        os.makedirs(os.path.join('models_resgan', fname))

        plot_model = {"epoch": [], "d_loss": [], "d_accuracy": [], "g_loss": []}

        x1_train = x1_train_df.values
        x2_train = x2_train_df.values

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

            plot_model["epoch"].append(epoch)
            plot_model["d_loss"].append(d_loss[0])
            plot_model["d_accuracy"].append(d_loss[1])
            plot_model["g_loss"].append(g_loss)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print('generating plots and saving outputs')
                gx1 = self.generator.predict(x1_train_df)
                self.generator.save(os.path.join('models', fname, 'generator' + str(epoch) + '.csv'))
                save_info.save_dataframes(epoch, x1_train_df, x2_train_df, gx1, fname, dir_name='output_resgan')
                save_info.save_scores(epoch, x1_train_df, x2_train_df, gx1, fname, dir_name='output_resgan')
                save_plots.plot_progress(epoch, x1_train_df, x2_train_df, gx1, plot_model, fname, umap=False,
                                         dir_name='figures_resgan', autoencoder=False, modelname='resgan')


if __name__ == '__main__':
    import os
    from loading_and_preprocessing.data_loader import load_data_basic, load_data_cytof
    path = r'C:\Users\heida\Documents\ETH\Deep Learning\chevrier_data_pooled_panels.parquet'
    x1_train, x1_test, x2_train, x2_test = load_data_basic(path, sample='sample5', batch_names=['batch1', 'batch2'],
                                                           seed=42, panel='tcell')
    # x1_train, x1_test, x2_train, x2_test = load_data_cytof(path, patient_id='rcc7', n=10000)

    # path = r'C:\Users\Public\PycharmProjects\deep\2019_DL_Class\loading_and_preprocessing'
    # path = path + '/toy_data_gamma_small.parquet'  # '/toy_data_gamma_large.parquet'
    # x1_train, x1_test, x2_train, x2_test = load_data_basic(path, patient='sample1', batch_names=['batch1', 'batch2'],
    # seed=42, n_cells_to_select=0)
    gan = GAN(x1_train.shape[1])
    gan.train(x1_train, x2_train, epochs=3000, batch_size=64, sample_interval=50)
