import numpy as np
import tensorflow as tf

''' Autoencoder with class inheritence  '''


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.sigmoid
        )

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            activation=tf.nn.sigmoid
        )

    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)


class Generator(tf.keras.Model): # Remark, the name of the model class is automatically set
    def __init__(self, intermediate_dim, original_dim):
        super(Generator, self).__init__()
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

    def call(self, input_features1, input_features2):
        code1 = self.encoder(input_features1)
        code2 = self.encoder(input_features2)
        reconstructed1 = self.decoder(code1)
        reconstructed2 = self.decoder(code2)
        return reconstructed1, code1, reconstructed2, code2

    def predict(self,input_features1, input_features2):
        code1 = self.encoder(input_features1)
        code2 = self.encoder(input_features2)
        reconstructed1 = self.decoder(code1)
        reconstructed2 = self.decoder(code2)
        return reconstructed1, code1, reconstructed2, code2


    def latent_space(self, input_features):
        code = self.encoder(input_features)
        return code


#    def call(self, input_features):
#        code = self.encoder(input_features)
#        reconstructed = self.decoder(code)
#        return reconstructed, code