from __future__ import print_function, division

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
#from keras.layers import *
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Constant


class LossWeighter(tf.keras.layers.Layer):
    def __init__(self, **kwargs):  # kwargs can have 'name' and other things
        super(LossWeighter, self).__init__(**kwargs)

    # create the trainable weight here, notice the constraint between 0 and 1
    def build(self, inputShape):
        self.weight = 0.05 # TODO change here
        super(LossWeighter, self).build(inputShape)

    def call(self, inputs):
        firstLoss, secondLoss = inputs
        return ( 0.5 * firstLoss) + (0.5 * secondLoss) + self.weight

    #def compute_output_shape(self, inputShape):
     #   return inputShape[0]
