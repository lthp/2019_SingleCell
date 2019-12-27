
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy

import numpy as np
import tensorflow as tf

a = tf.Tensor([[ 2 , 6], [12, 20]])


bar = x1[1:10, 1:20]
baz = x1_train[1:12, 1:20]
sigmas = [1e-6, 1e-5]
s = tf.constant(sigmas, dtype='float64')
maximum_mean_discrepancy(bar, baz,s )

x1
x1l = tf.constant(x1_labels)
idx1 = tf.where(equal(x1l, 15))
mask = tf.constant(np.zeros(shape = x1.shape), dtype = bool) #shape=(64, 1066),
idx11 = tf.reshape(idx1, [-1])



x1l = tf.constant(x1_labels)
foo1 = tf.constant(x1)
idx1 = tf.where(equal(x1l, 15))
mask = None
for i in np.arange(x1.shape[0]):
  if i in idx1:
    extend = tf.expand_dims(tf.constant(np.ones(shape=x1.shape[1]), dtype=bool), 1)
  else:
    extend = tf.expand_dims(tf.constant(np.zeros(shape=x1.shape[1]), dtype=bool), 1)
  if mask is not None:
   mask = tf.concat([mask, extend], axis = 1)
  else:
    mask = extend
mask = tf.transpose(mask)






