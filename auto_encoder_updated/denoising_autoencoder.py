from keras import callbacks as cb
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np

from auto_encoder_updated import data_hander as dh, monitoring as mn
import os.path


class Sample:
    x = None
    y = None

    def __init__(self, x, y=None):
        self.x = x
        self.y = y


def trainDAE(target, data_path, ref_sample_ind, train_index, relevant_markers, mode,
             keep_prob, denoise, load_model, path):
    source_x = []
    for i in np.arange(train_index.size - 1):
        source_index = np.delete(train_index, ref_sample_ind)[i]
        source = dh.load_data(data_path, source_index, relevant_markers, mode)
        num_zeros_ok = 1
        to_keep_s = np.sum((source.x == 0), axis=1) <= num_zeros_ok
        if i == 0:
            source_x = source.x[to_keep_s]
        else:
            source_x = np.concatenate([source_x, source.x[to_keep_s]], axis=0)

    # preProcess source
    source_x = np.log(1 + np.abs(source_x))

    num_zeros_ok = 1
    to_keep_t = np.sum((target.x == 0), axis=1) <= num_zeros_ok

    input_dim = target.x.shape[1]

    ae_encoding_dim = 25
    l2_penalty_ae = 1e-2

    if denoise:
        if load_model:
            from keras.models import load_model
            autoencoder = load_model(os.path.join(os.path.dirname(__file__), 'savemodels/' + path + '/denoisedAE.h5'))
        else:
            # train de-noising auto encoder and save it.
            train_target_ae = np.concatenate([source_x, target.x[to_keep_t]],
                                             axis=0)
            train_data_ae = train_target_ae * np.random.binomial(n=1, p=keep_prob,
                                                                 size=train_target_ae.shape)

            input_cell = Input(shape=(input_dim,))
            encoded = Dense(ae_encoding_dim, activation='relu',
                            kernel_regularizer=l2(l2_penalty_ae))(input_cell)
            encoded1 = Dense(ae_encoding_dim, activation='relu',
                             kernel_regularizer=l2(l2_penalty_ae))(encoded)
            decoded = Dense(input_dim, activation='linear',
                            kernel_regularizer=l2(l2_penalty_ae))(encoded1)

            autoencoder = Model(inputs=input_cell, outputs=decoded)
            autoencoder.compile(optimizer='rmsprop', loss='mse')
            autoencoder.fit(train_data_ae, train_target_ae, epochs=80,
                            batch_size=128, shuffle=True,
                            validation_split=0.1, verbose=0,
                            callbacks=[mn.monitor(), cb.EarlyStopping(
                                monitor='val_loss', patience=25, mode='auto')])
            autoencoder.save(os.path.join(os.path.dirname(__file__), 'savemodels/' + path + '/denoisedAE.h5'))
            del source_x
            plt.close('all')

        return autoencoder


def predictDAE(target, autoencoder, denoise=False):
    if denoise:
        # apply de-noising auto encoder to target.
        denoise_target = Sample(autoencoder.predict(target.x), target.y)
    else:
        denoise_target = Sample(target.x, target.y)

    return denoise_target
