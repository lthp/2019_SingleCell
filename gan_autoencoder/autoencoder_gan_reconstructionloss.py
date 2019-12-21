from __future__ import print_function, division

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualisation_and_evaluation.helpers_vizualisation import plot_tsne, plot_metrics, plot_umap
from datetime import datetime
from visualisation_and_evaluation.helpers_eval import evaluate_scores, separate_metadata, prep_data_for_eval

'''
This model is an optimized gan where the generator is an autoencoder with reconstruction loss, but the structure of 
the generator autoencoder is diamond shaped (not with a bottleneck layer). This model seems to be performing good.
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
        loss_weights = {'generator': 1,
                        'discriminator': 0.1}
        metrics = {'discriminator': 'accuracy'}
        self.combined.compile(loss=losses, optimizer=optimizer, loss_weights=loss_weights, metrics=metrics)


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
        os.makedirs(os.path.join('figures', fname))
        os.makedirs(os.path.join('output', fname))
        os.makedirs(os.path.join('models', fname))

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
                if d_loss[1] > 0.8:      # Gives the generator a break if the discriminator learns too fast
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
            #g_loss = np.mean(g_loss_list, axis=0)
            #d_loss = np.mean(d_loss_list, axis=0)
            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, mae: %.2f, xentropy: %f, acc.: %.2f%%]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], g_loss[2], g_loss[3]*100))

            plot_model["epoch"].append(epoch)
            plot_model["d_loss"].append(d_loss[0])
            plot_model["g_loss"].append(g_loss[2])

            plot_model["d_accuracy"].append(d_loss[1])
            plot_model["g_accuracy"].append(g_loss[3])

            plot_model["g_reconstruction_error"].append(g_loss[1])
            plot_model["g_loss_total"].append(g_loss[0])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print('generating plots and saving outputs')
                gx1 = self.generator.predict(x1_train_df)
                self.generator.save(os.path.join('models', fname, 'generator' + str(epoch) + '.csv'))
                # self.plot_progress(epoch, x1_train_df, x2_train_df, gx1, plot_model, fname)
                self.save_info(epoch, x1_train_df, x2_train_df, gx1, fname)
        return plot_model

    def transform_batch(self, x):
        gx = self.generator.predict(x)
        gx_df = pd.DataFrame(data=gx, columns=x.columns, index=x.index + '_transformed')
        return gx_df

    def plot_progress(self, epoch, x1, x2, gx1, metrics, fname):
        plot_metrics(metrics, os.path.join('figures', fname, 'metrics'), autoencoder=True)
        if epoch == 0:
            plot_tsne(pd.concat([x1, x2]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                      save_as=os.path.join(fname, 'aegan_tsne_x1-x2_epoch'+str(epoch)))
            plot_umap(pd.concat([x1, x2]), save_as=os.path.join(fname, 'aegan_umap_x1-x2_epoch'+str(epoch)))

        gx1 = pd.DataFrame(data=gx1, columns=x1.columns, index=x1.index + '_transformed')
        plot_tsne(pd.concat([x1, gx1]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                  save_as=os.path.join(fname, 'aegan_tsne_x1-gx1_epoch'+str(epoch)))
        plot_tsne(pd.concat([gx1, x2]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                  save_as=os.path.join(fname, 'aegan_tsne_gx1-x2_epoch'+str(epoch)))
        plot_umap(pd.concat([x1, gx1]), save_as=os.path.join(fname, 'aegan_umap_gx1-x1_epoch'+str(epoch)))
        plot_umap(pd.concat([x2, gx1]), save_as=os.path.join(fname, 'aegan_umap_gx1-x2_epoch'+str(epoch)))

    def save_info(self, epoch, x1, x2, gx1, fname):
        if epoch == 0:
            x1.to_csv(os.path.join('output', fname, 'x1_epoch' + str(epoch) + '.csv'),
                      index_label=False)
            x2.to_csv(os.path.join('output', fname, 'x2_epoch' + str(epoch) + '.csv'),
                      index_label=False)

        gx1 = pd.DataFrame(data=gx1, columns=x1.columns, index=x1.index + '_transformed')
        df_eval = pd.concat([gx1, x2])
        df, metadf = separate_metadata(df_eval)
        umap_codes, data, cell_type_labels, batch_labels, num_datasets = prep_data_for_eval(df, metadf, 50,
                                                                                            random_state=345)
        divergence_score, entropy_score, silhouette_score = evaluate_scores(umap_codes, data, cell_type_labels,
                                                                            batch_labels, num_datasets,
                                                                            50, 50, 'cosine',
                                                                            random_state=345)

        f = open(os.path.join('output', fname, 'div_score.csv'), 'a', newline='')
        score_writer = csv.writer(f)
        score_writer.writerow([epoch, divergence_score])
        print(divergence_score, entropy_score, silhouette_score)
        gx1.to_csv(os.path.join('output', fname, 'gx1_epoch' + str(epoch) + '.csv'),
                   index_label=False)



if __name__ == '__main__':
    import os
    from loading_and_preprocessing.data_loader import load_data_basic, load_data_cytof
    # path = r'C:\Users\heida\Documents\ETH\Deep Learning\2019_DL_Class_old\code_ADAE_\chevrier_data_pooled_panels.parquet'
    # path = r'C:\Users\Public\PycharmProjects\deep\Legacy_2019_DL_Class\data\chevrier_data_pooled_panels.parquet'
    # x1_train, x1_test, x2_train, x2_test = load_data_cytof(path, patient_id='rcc7', n=10000)

    path = r'C:\Users\Public\PycharmProjects\deep\2019_DL_Class\loading_and_preprocessing'
    path = path + '/toy_data_gamma_small.parquet'  # '/toy_data_gamma_large.parquet'
    x1_train, x1_test, x2_train, x2_test = load_data_basic(path, patient='sample1', batch_names=['batch1', 'batch2'],
                                                           seed=42, n_cells_to_select=0)
    gan = GAN(x1_train.shape[1])
    gan.train(x1_train, x2_train, epochs=3000, batch_size=64, sample_interval=50)
