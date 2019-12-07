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
    def __init__(self):
        self.data_size = 30
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
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(20, input_dim=self.data_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(15))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.data_size, activation='tanh'))
        # model.add(Reshape(self.img_shape))
        model.summary()

        x1 = Input(shape=(self.data_size,))
        x1_gen = model(x1)

        return Model(x1, x1_gen)

    def build_generator2(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.data_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(112))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.data_size, activation='tanh'))
        # model.add(Reshape(self.img_shape))
        model.summary()

        x1 = Input(shape=(self.data_size,))
        x1_gen = model(x1)

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.data_size,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        x2 = Input(shape=self.data_size)
        validity = model(x2)
        return Model(x2, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        path = '/Users/Public/PycharmProjects/deep/2019_DL_Class/data/chevrier_data_pooled_panels.parquet'
        full = pd.read_parquet(path, engine='pyarrow')
        select_cols = [col for col in full.columns if not "metadata" in col]  # not include metadata
        select_cols.append('metadata_panel')
        full = full.loc[:, select_cols]
        panels = full.metadata_panel.unique()
        full_panel1 = full.loc[full['metadata_panel'] == panels[0]]
        full_panel2 = full.loc[full['metadata_panel'] == panels[1]]

        # start working with batches and patients in panel1 only
        full_panel1 = full_panel1.dropna(how='all', axis='columns')
        full_panel1 = full_panel1.loc[:, ]
        full_patient = full_panel1.reset_index()
        full_patient = full_patient.rename({'level_0': 'batch', 'level_1': 'patient', 'level_2': 'cell'}, axis=1)
        ID = 'rcc7'  # start with only 1 patient
        full_patient = full_patient.loc[full_patient['patient'] == ID, :]
        batches = full_patient.batch.unique()
        full_patient_batch1 = full_patient.loc[full_patient['batch'] == batches[0]]  # split into the 2 batches
        full_patient_batch2 = full_patient.loc[full_patient['batch'] == batches[1]]  # for this patient
        full_patient_batch1 = full_patient_batch1.iloc[1:1204, :]  # batch 1 has otherwise much more number of cells
        full_patient_batch2 = full_patient_batch2.iloc[1:1204, :]

        # y = full_patient_batch1["batch"]  # the label is batch1 (the reference batch)
        x1 = full_patient_batch1.drop(["batch", 'cell', 'patient', 'metadata_panel'],
                                      axis=1)  # remove all but the markers as the data
        x2 = full_patient_batch2.drop(["batch", 'cell', 'patient', 'metadata_panel'], axis=1)
        # x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, test_size=0.33,
                                                                # random_state=42)  # split into train and test
        x1 = x1.values
        x2 = x2.values

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
            gen_x1 = self.generator.predict(x1)

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
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_x2(epoch, x1)

    def sample_x2(self, epoch, x1):
        r, c = 5, 5
        gen_x2 = self.generator.predict(x1)
        # can e.g. save some pca figures of the gen_x2


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=64, sample_interval=200)