'''Inspired from this https://github.com/eriklindernoren/Keras-GAN and Bermuda paper https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1764-6 '''
from __future__ import print_function, division

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import sys

sys.path.append("..")
sys.path.append("/cluster/home/prelotla/GitHub/projects2019_DL_Class")

from helpers_bermuda import pre_processing, read_cluster_similarity, make_mask_tensor
from AE_bermuda import Autoencoder
from MMD_bermuda import maximum_mean_discrepancy
from info_on_checkpoint import save_info_bermuda, save_plots



'''
This model is an optimized gan where the generator is an autoencoder with reconstruction loss, and the structure of 
the generator autoencoder is hour-glass shaped ( with a bottleneck layer) and has batch norm layers. 
This model seems to be performing good.
'''

seed = 12345

class GAN():
    def __init__(self, n_markers=30, cluster_pairs = None, n_clusters = None ) :
        self.data_size = n_markers
        self.cluster_pairs = cluster_pairs
        self.optimizer = Adam(0.0002, 0.5)
        sigmas = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6]
        sigmas = tf.constant(sigmas, dtype = 'float64')
        self.sigmas =(tf.expand_dims(sigmas, 1))
        self.intermed_dim = 40
        self.n_clusters = n_clusters

        x1 = Input(shape=(self.data_size,), name = 'x1')
        x2 = Input(shape=(self.data_size,), name = 'x2')
        mask_clusters = Input(shape = (None, self.n_clusters + 1 ), dtype = 'float64')

        # Build the simple autoencoder1
        x = x1
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(self.intermed_dim)(x)
        x = LeakyReLU(alpha=0.2, name = 'code1_layer')(x)
        code1 = x

        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(self.data_size)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Activation('tanh', name = 'autoencoder_x1')(x)
        gen_x1 = x

        # Build the simple autoencoder2
        x = x2
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(self.intermed_dim)(x)
        x = LeakyReLU(alpha=0.2, name = 'code2_layer')(x)
        code2 = x

        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(self.data_size)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Activation('tanh', name = 'autoencoder_x2')(x)
        gen_x2 = x


        def calculate_reconstruction_loss(y_true, y_pred):
            r_cost = tf.keras.losses.MSE(y_true, y_pred)
            return r_cost

        def reconstruction_loss(x1, gen_x1):
            Loss =  calculate_reconstruction_loss(gen_x1, x1) + calculate_reconstruction_loss(gen_x2, x2) #TODO AM I USING THIS GUY?????
            return Loss

        def transfert_loss(x1, gen_x1):
            Loss = 0
            for i, row in enumerate(self.cluster_pairs):
                cluster_idx1 = np.int(row[1])
                cluster_idx2 = np.int(row[0])
                mask_clusters_1 = Lambda(lambda x: x[:, :, cluster_idx1])(mask_clusters)
                mask_clusters_2 = Lambda(lambda x: x[:, :, cluster_idx2])(mask_clusters)
                code1_single =  Lambda(lambda x: tf.transpose(tf.matmul(tf.transpose(x), mask_clusters_1)),
                                           name = 'onecluster_code1')(code1)
                code2_single = Lambda(lambda x: tf.transpose(tf.matmul(tf.transpose(x), mask_clusters_2)),
                                          name='onecluster_code1')(code2)
                add_ = maximum_mean_discrepancy(code1_single, code2_single, self.sigmas)
                Loss = tf.math.add(add_, Loss)

            return Loss


        def autoencoder_loss(x1, gen_x1):
            Loss = transfert_loss(x1, gen_x1) + reconstruction_loss(x1, gen_x1)
            return Loss

        self.fullGenerator = Model(inputs=[x1, x2, mask_clusters], outputs=gen_x1, name = 'fullGenerator')
        #self.fullGenerator.compile(optimizer=self.optimizer, loss= autoencoder_loss ,
        #                            experimental_run_tf_function=False,
        #                            metrics=[transfert_loss,
        #                                     reconstruction_loss]) # experimental_run_tf_function explained in https://github.com/tensorflow/probability/issues/519

        #Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        # Build combined model
        self.discriminator.trainable = False
        # The discriminator takes generated data as input and determines validity

        outputs = self.discriminator(self.fullGenerator([x1, x2, mask_clusters]))
        #validity = self.discriminator(gen_x1)

        self.combined = Model(inputs=[x1, x2, mask_clusters], outputs= [gen_x1, outputs]) #passes the gen_x1 output into validity

        losses = {'autoencoder_x1': autoencoder_loss,
                  'discriminator': 'binary_crossentropy'}
        loss_weights = {'autoencoder_x1': 0.80,
                        'discriminator': 0.20}
        metrics = {'discriminator': 'accuracy',
                   'autoencoder_x1': [transfert_loss,
                                            reconstruction_loss]}
        self.combined.compile(loss=losses, optimizer=self.optimizer,
                              loss_weights=loss_weights, metrics=metrics,
                              experimental_run_tf_function=False)

    ##############################################


    def build_generator(self):
        model = Autoencoder(20, self.data_size )
        return model

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
####################


    def train(self, x1_train_df, x2_train_df, epochs, batch_size=128, sample_interval=50, cell_label_path = None, output_path = None ):
        cell_label = pd.read_table(cell_label_path, sep = '\t')

        fname = datetime.now().strftime("%d-%m-%Y_%H.%M.%S")
        figures_ = os.path.join(output_path, 'outputs_figures')
        frames_ = os.path.join(output_path, 'output_dataframes')
        models_ = os.path.join(output_path, 'output_models')
        scores_ = os.path.join(output_path, 'output_scores')
        for save_type in [figures_, frames_, scores_, models_]:
            os.makedirs(os.path.join(save_type, fname))


        plot_model = {
            'epoch': [],
            'Combined: loss': [],
            'Combined: autoencoder_x1_loss': [],
            'Combined: discriminator_loss': [],
            'Combined: autoencoder_x1_transfert_loss': [],
            'Combined: autoencoder_x1_reconstruction_loss': [],
            'Combined: discriminator_accuracy': [],
            'Discriminator: loss': [],
            'Discriminator: accuracy': []
        }



        x1_train = x1_train_df['gene_exp'].transpose()
        x2_train = x2_train_df['gene_exp'].transpose()
        x1_labels = x1_train_df['cluster_labels']
        x2_labels = x2_train_df['cluster_labels']

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))  # TODO check: assume normalisation between zero and 1
        fake = np.zeros((batch_size, 1))

        valid_full = np.ones((len(x1_train), 1))
        fake_full = np.zeros((len(x1_train), 1))
        d_loss = [0, 0]

        steps_per_epoch = max(len(x1_train), len(x2_train)) // batch_size
        percent = batch_size / x1_train.shape[0]
        sss = StratifiedShuffleSplit(n_splits=steps_per_epoch, train_size=percent, random_state=12345)
        for epoch in range(epochs):
            d_loss_list = []
            g_loss_list = []
            train_generator_x1 = sss.split(x1_train, x1_labels)
            train_generator_x2 = sss.split(x2_train, x2_labels)
            for (gener_idx1, gener_idx2) in zip(train_generator_x1, train_generator_x2):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of x1 and x2
                idx1 = gener_idx1[0]
                idx2 = gener_idx2[0]
                x1 = x1_train[idx1]
                x2 = x2_train[idx2]
                x1_lab = x1_labels[idx1]
                x2_lab = x2_labels[idx2]
                mask_clusters = make_mask_tensor(x1, x2, x1_lab, x2_lab)
                assert(x1.shape[0] ==  batch_size)
                assert (x2.shape[0] == batch_size)

                # Generate a batch of new images
                gen_x1 = self.fullGenerator.predict([x1, x2, mask_clusters])


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
                g_loss = self.combined.train_on_batch([x1, x2, mask_clusters], [x1, valid]) #TODO Add the generator loss with latent space, inside or outside?? Need generator with two inputs and custom loss (First = take just the suum of the losses


                g_loss_list.append(g_loss)
                d_loss_list.append(d_loss)

            print('\n')
            print('epoch {} start'.format(epoch))
            mask_clusters = make_mask_tensor(x1_train, x2_train, x1_labels, x2_labels)
            print('made mask')
            gen_x1 = self.fullGenerator.predict([x1_train, x2_train, mask_clusters])
            print('made generator predict')
            g_loss = self.combined.test_on_batch([x1_train, x2_train, mask_clusters], [x1_train, valid_full]) #
            print('made combined test on batch ')
            d_loss = self.discriminator.test_on_batch(np.concatenate((x2_train, gen_x1)),
                                                      np.concatenate((valid_full, fake_full)))
            print('made discriminator test on batch')

            # Plot the progress
            #print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, mae: %.2f, xentropy: %f, acc.: %.2f%%]" %
            #      (epoch, d_loss[0], 100 * d_loss[1],
            #       g_loss[0], g_loss[1], g_loss[2], g_loss[3] * 100))

            for value, item in zip(g_loss, self.combined.metrics_names):
                print("Combined: {} = {}".format(item, value))
                plot_model["Combined: {}".format(item)].append(value)

            for value, item in zip(d_loss, self.discriminator.metrics_names):
                print("Discriminator: {} = {}".format(item, value))
                plot_model["Discriminator: {}".format(item)].append(value)
            plot_model["epoch"].append(epoch)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print('generating plots and saving outputs')
                # gen_x1 = self.generator.predict(x1_train_df)

                # self.generator.save(os.path.join('models', fname, 'generator' + str(epoch) + '.csv')) # save model
                save_info_bermuda.save_dataframes(epoch, x1_train, x2_train, gen_x1, fname, dir_name = frames_) # save dataframe
                save_info_bermuda.save_scores(epoch=epoch,
                                              x1=x1_train, x2=x2_train,
                                              metrics=plot_model, gx1=gen_x1,
                                              fname= fname,
                                              x1_cells = x1_train_df['cell_labels'],
                                              x2_cells = x2_train_df['cell_labels'],
                                              cell_label = cell_label,
                                              dir_name = scores_) # save metrics
                #save_plots.plot_progress(epoch, x1_train_df, x2_train_df, gx1, plot_model, fname, umap=False) # plots

        return plot_model



if __name__ == '__main__':
    import os
    path = os.getcwd()

    # IMPORTANT PARAMETER
    similarity_thr = 0.90  # S_thr in the paper, choose between 0.85-0.9

    pre_process_paras = {'take_log': False, 'standardization': False, 'scaling': False, 'oversample': True, 'split':0.80, 'separator':'\t', 'reduce_set' : 5} # TODO Change reduce set
    base_dir = '/cluster/work/grlab/projects/tmp_laurie/dl_data'
    #base_dir = '/Users/laurieprelot/Documents/Projects/2019_Deep_learning/data/Chevrier-et-al'
    path_data1_clusters = os.path.join(base_dir, 'normalized', 'chevrier_data_pooled_full_panels.batch3.bermuda.tsv')
    path_data2_clusters = os.path.join(base_dir, 'normalized', 'chevrier_data_pooled_full_panels.batch1.bermuda.tsv')
    cluster_similarity_file = os.path.join(base_dir, 'metaneighbor', 'chevrier_data_pooled_full_panels.batch1_batch3.bermuda_metaneighbor_subsample.tsv')
    equiv_table_path_cells = os.path.join(base_dir, 'equivalence_tables', 'equivalence_table_metadata_celltype.tsv')
    output_path = os.path.join(base_dir, 'Outputs')

    dataset_file_list = [path_data1_clusters, path_data2_clusters]
    cluster_pairs = read_cluster_similarity(cluster_similarity_file, similarity_thr , pre_process_paras['separator'])
    x1_train, x1_test, x2_train, x2_test = pre_processing(dataset_file_list, pre_process_paras)
    n_clusters = max(np.concatenate( [x1_train['cluster_labels'], x2_train['cluster_labels'] ]))

    gan = GAN(len(x1_train['gene_sym']), cluster_pairs, n_clusters = n_clusters ) #
    gan.train(x1_train, x2_train, epochs=1000, batch_size=40, sample_interval=1, cell_label_path = equiv_table_path_cells, output_path = output_path)#TODO change sample intervals


