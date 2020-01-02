from __future__ import print_function, division

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from info_on_checkpoint import save_info, save_plots
import numpy as np
from datetime import datetime


'''
This model is an optimized gan where the generator is an autoencoder with reconstruction loss, and the structure of 
the generator autoencoder is hour-glass shaped ( with a bottleneck layer) and has batch norm layers. 
This model seems to be performing good.
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
        self.combined = Model(inputs=x1, outputs=[gen_x1, validity])
        losses = {'generator': 'mean_absolute_error',
                  'discriminator': 'binary_crossentropy'}
        loss_weights = {'generator': 0.8,
                        'discriminator': 0.2}
        metrics = {'discriminator': 'accuracy'}
        self.combined.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights, metrics=metrics)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.data_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(15))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.data_size, activation='tanh'))
        model.summary()

        x1 = Input(shape=(self.data_size,))
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
        fname = datetime.now().strftime("%d-%m-%Y_%H.%M.%S")
        # fname = '_ganautobottlebatch_loss0.8_full_corrupsample' + x1_train_df.index[0].split('.')[0]
        # fname = time + fname
        os.makedirs(os.path.join('figures_bottleneck', fname))
        os.makedirs(os.path.join('output_bottleneck', fname))
        os.makedirs(os.path.join('models_bottleneck', fname))

        training_metrics = {"epoch": [], "d_loss": [], "d_accuracy": [], "g_loss": []}

        plot_model = {"epoch": [], "d_loss": [], "g_loss": [], "d_accuracy": [], "g_accuracy": [],
                      "g_reconstruction_error": [], "g_loss_total": []}

        x1_train = x1_train_df.values
        x2_train = x2_train_df.values

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        valid_full = np.ones((len(x1_train), 1))
        fake_full = np.zeros((len(x1_train), 1))
        d_loss = [0, 0]

        steps_per_epoch = len(x1_train) // batch_size
        for epoch in range(epochs):
            d_loss_list = []
            g_loss_list = []
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
            g_loss = self.combined.test_on_batch(x1_train, [x1_train, valid_full])
            d_loss = self.discriminator.test_on_batch(np.concatenate((x2_train, gen_x1)),
                                                      np.concatenate((valid_full, fake_full)))
            # g_loss = np.mean(g_loss_list, axis=0)
            # d_loss = np.mean(d_loss_list, axis=0)
            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, mae: %.2f, xentropy: %f, acc.: %.2f%%]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1], g_loss[2], g_loss[3] * 100))

            plot_model["epoch"].append(epoch)
            plot_model["d_loss"].append(d_loss[0])
            plot_model["g_loss"].append(g_loss[2])

            plot_model["d_accuracy"].append(d_loss[1])
            plot_model["g_accuracy"].append(g_loss[3])

            plot_model["g_reconstruction_error"].append(g_loss[1])
            plot_model["g_loss_total"].append(g_loss[0])

            training_metrics["epoch"].append(epoch)
            training_metrics["d_loss"].append(d_loss[0])
            training_metrics["d_accuracy"].append(d_loss[1])
            training_metrics["g_loss"].append(g_loss)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print('generating plots and saving outputs')
                gx1 = self.generator.predict(x1_train_df)
                self.generator.save(os.path.join('models_bottleneck', fname, 'generator' + str(epoch) + '.csv'))
                save_info.save_dataframes(epoch, x1_train_df, x2_train_df, gx1, fname, dir_name='output_bottleneck')
                save_info.save_scores(epoch, x1_train_df, x2_train_df, gx1, training_metrics, fname,
                                      dir_name='output_bottleneck')
                # save_plots.plot_progress(epoch, x1_train_df, x2_train_df, gx1, plot_model, fname, dir_name='figures_bottleneck')

        return plot_model


if __name__ == '__main__':
    import os
    from loading_and_preprocessing.data_loader import load_data_basic, load_data_cytof

    # path = r'C:\Users\heida\Documents\ETH\Deep Learning\2019_DL_Class_old\code_ADAE_\chevrier_data_pooled_panels.parquet'
    # path = r'C:\Users\Public\PycharmProjects\deep\Legacy_2019_DL_Class\data\chevrier_data_pooled_panels.parquet'

    path = r'C:\Users\Public\PycharmProjects\deep\Legacy_2019_DL_Class\data\chevrier_data_pooled_full_panels.parquet'
    x1_train, x1_test, x2_train, x2_test = load_data_basic(path, sample='sample75', batch_names=['batch1', 'batch3'],
                                                           seed=42, panel=None)
    # x1_train, x1_test, x2_train, x2_test = load_data_cytof(path, patient_id='rcc7', n=10000)

    # path = os.getcwd()
    # path = path + '/toy_data_gamma_small.parquet'  # '/toy_data_gamma_large.parquet'
    # x1_train, x1_test, x2_train, x2_test = load_data_basic(path, patient='sample1', batch_names=['batch1', 'batch2'],
    #                                                       seed=42, n_cells_to_select=0)
    gan = GAN(x1_train.shape[1])
    gan.train(x1_train, x2_train, epochs=1000, batch_size=64, sample_interval=50)
