# !/usr/bin/env python
from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

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

'''
This model is an optimized gan where the generator is an autoencoder with reconstruction loss, but the structure of 
the generator autoencoder is diamond shaped (not with a bottleneck layer) AND has batch norm layers.
'''


class GAN():
    def __init__(self, modelname, n_markers=30, loss_lambda=0.9):
        self.modelname = modelname
        self.data_size = n_markers
        self.loss_lambda = loss_lambda
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
        losses = {'generator': 'mean_absolute_error',
                  'discriminator': 'binary_crossentropy'}

        loss_weights = {'generator': self.loss_lambda,
                        'discriminator': 1-self.loss_lambda}
        metrics = {'discriminator': 'accuracy'}
        self.combined.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights, metrics=metrics)

    def residual_block(self, number_of_nodes):
        x1 = Input(shape=(self.data_size,))
        x = x1
        for n in number_of_nodes:
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dense(n)(x)

        # x = Dense(self.data_size)(x)
        x = Activation('tanh')(x)

        x = Add()([x, x1])
        return Model(x1, x)

    def build_generator(self):
        x1 = Input(shape=(self.data_size,))

        block1 = self.residual_block([30, 40, self.data_size])(x1)
        block2 = self.residual_block([30, 40, self.data_size])(block1)
        block3 = self.residual_block([30, 40, self.data_size])(block2)

        # model.add(Reshape(self.img_shape))

        model = Model(x1, block3)
        model.summary()

        x1_gen = model(x1)

        return Model(x1, x1_gen, name='generator')

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
        return Model(x2, validity, name='discriminator')

    def train(self, x1_train_df, x2_train_df, epochs, batch_size=128, sample_interval=50):
        time = datetime.now().strftime("%d-%m-%Y_%H.%M.%S")
        model_description = '_' + self.modelname + '_lambda' + str(self.loss_lambda) + '_' + \
                            x1_train_df.index[0].split('_')[1]
        fname = time + model_description
        os.makedirs(os.path.join('figures_' + self.modelname, fname))
        os.makedirs(os.path.join('output_' + self.modelname, fname))
        os.makedirs(os.path.join('models_' + self.modelname, fname))

        training_metrics = {"epoch": [], "d_loss": [], "g_loss": [], "d_accuracy": [], "g_accuracy": [],
                            "g_reconstruction_error": [], "g_loss_total": []}

        x1_train = x1_train_df.values
        x2_train = x2_train_df.values

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        valid_full_x1 = np.ones((len(x1_train), 1))
        fake_full_x1 = np.zeros((len(x1_train), 1))
        valid_full_x2 = np.ones((len(x2_train), 1))
        d_loss = [0, 0]

        steps_per_epoch = np.max([x1_train.shape[0], x2_train.shape[0]]) // batch_size
        for epoch in range(epochs):
            d_loss_list = []
            g_loss_list = []
            for step in range(steps_per_epoch):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of x1 and x2
                idx1 = np.random.randint(0, x1_train.shape[0], batch_size)
                x1 = x1_train[idx1]
                idx2 = np.random.randint(0, x2_train.shape[0], batch_size)
                x2 = x2_train[idx2]

                # Generate a batch of new images
                gen_x1 = self.generator.predict(x1)

                # Train the discriminator
                if d_loss[1] > 0.8:  # Gives the generator a break if the discriminator learns too fast
                    d_loss_real = self.discriminator.test_on_batch(x2, valid)
                    d_loss_fake = self.discriminator.test_on_batch(gen_x1, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                else:
                    d_loss_real = self.discriminator.train_on_batch(x2, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_x1, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(x1, [x1, valid])

                g_loss_list.append(g_loss)
                d_loss_list.append(d_loss)

            gen_x1 = self.generator.predict(x1_train)
            g_loss = self.combined.test_on_batch(x1_train, [x1_train, valid_full_x1])
            d_loss = self.discriminator.test_on_batch(np.concatenate((x2_train, gen_x1)),
                                                      np.concatenate((valid_full_x2, fake_full_x1)))
            # g_loss = np.mean(g_loss_list, axis=0)
            # d_loss = np.mean(d_loss_list, axis=0)
            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, mae: %.2f, xentropy: %f, acc.: %.2f%%]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1], g_loss[2], g_loss[3] * 100), flush=True)

            training_metrics["epoch"].append(epoch)
            training_metrics["d_loss"].append(d_loss[0])
            training_metrics["g_loss"].append(g_loss[2])

            training_metrics["d_accuracy"].append(d_loss[1])
            training_metrics["g_accuracy"].append(g_loss[3])

            training_metrics["g_reconstruction_error"].append(g_loss[1])
            training_metrics["g_loss_total"].append(g_loss[0])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print('generating plots and saving outputs')
                gx1 = self.generator.predict(x1_train_df)
                self.generator.save(os.path.join('models_' + self.modelname, fname, 'generator' + str(epoch)))
                save_info.save_dataframes(epoch, x1_train_df, x2_train_df, gx1, fname,
                                          dir_name='output_'+self.modelname)
                save_info.save_scores(epoch, x1_train_df, x2_train_df, gx1, training_metrics, fname,
                                      dir_name='output_'+self.modelname, model_description=model_description)
                #save_plots.plot_progress(epoch, x1_train_df, x2_train_df, gx1, training_metrics, fname, umap=True,
                #                         dir_name='figures_'+self.modelname, autoencoder=True, modelname=self.modelname)


if __name__ == '__main__':
    import os
    from loading_and_preprocessing.data_loader import load_data_basic
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('loss_lambda', type=float)
    args = parser.parse_args()
    sample_name = 'sample' + args.sample
    x1_train, x1_test, x2_train, x2_test = load_data_basic(args.path, sample=sample_name,
                                                           batch_names=['batch1', 'batch3'], seed=42, panel=None,
                                                           upsample=True)
    gan = GAN('residual_gan_full_panels', x1_train.shape[1], args.loss_lambda)
    gan.train(x1_train, x2_train, epochs=1000, batch_size=64, sample_interval=50)
