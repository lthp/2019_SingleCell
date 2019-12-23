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
from tensorflow.keras.backend import equal, sum
import numpy as np


class LossWeighter(tf.keras.layers.Layer):
    def __init__(self, code1, cluster_labels1, code2, cluster_labels2, cluster_pairs):  # kwargs can have 'name' and other things
        super(LossWeighter, self).__init__() #**kwargs
        self.code1 =  code1
        self.code2 = code2
        self.cluster_labels1 = cluster_labels1
        self.cluster_labels2 = cluster_labels2
        self.cluster_pairs  = cluster_pairs
        #self.name='weighted_loss'
    #def build( self, inputShape, code1, cluster_labels1, code2, cluster_labels2):
    #    super(LossWeighter, self).build(inputShape)
    #    self.weight = cluster_labels2.shape[1] / 100000
        #self.weight = 0.05 # TODO change here


    def call(self, inputs):
        ''' firstLoss, secondLoss , code1, cluster_labels1, code2, cluster_labels2, cluster_pairs'''
        firstLoss, secondLoss =  inputs
        #test = tf.where(equal(self.cluster_labels1, 1.))
        #test_bis = tf.where(equal(cluster_labels1, 1.))
        self.weight = 0.5
        model_loss = ( 0.5 * firstLoss) + (0.5 * secondLoss) + self.weight
        return model_loss

    #def compute_output_shape(self, inputShape):
     #   return inputShape[0]
