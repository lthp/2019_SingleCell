
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



##### BACKUP###


from __future__ import print_function, division

import tensorflow as tf
from functools import partial
import numpy as np

"""Domain Adaptation Loss Functions.
The following domain adaptation loss functions are defined:
- Maximum Mean Discrepancy (MMD).
  Relevant paper:
    Gretton, Arthur, et al.,
    "A kernel two-sample test."
    The Journal of Machine Learning Research, 2012
Implementation from: https://github.com/tensorflow/models/blob/master/research/domain_adaptation/domain_separation/losses.py#L40 
"""


def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.
  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """
  # i = False
  # if not i:
  #   raise ValueError('Shape {}'.format(x.get_shape()))
  #
  # if not len(x.get_shape()) == len(y.get_shape()) == 2:
  #   raise ValueError('Both inputs should be matrices.')
  #
  # if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
  #   raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
  We create a _sum of multiple gaussian kernels_ each having a width sigma_i.
  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, sigmas, kernel=gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
      sigmas:a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x, sigmas))
    cost += tf.reduce_mean(kernel(y, y, sigmas))
    cost -= 2 * tf.reduce_mean(kernel(x, y, sigmas))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


# def mmd_loss(source_samples, target_samples, weight, scope=None):
#   """Adds a similarity loss term, the MMD between two representations.
#   This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
#   different Gaussian kernels.
#   Args:
#     source_samples: a tensor of shape [num_samples, num_features].
#     target_samples: a tensor of shape [num_samples, num_features].
#     weight: the weight of the MMD loss.
#     scope: optional name scope for summary tags.
#   Returns:
#     a scalar tensor representing the MMD loss value.
#   """
#   sigmas = [
#       1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
#       1e3, 1e4, 1e5, 1e6
#   ]
#   gaussian_kernel = partial(
#       gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
#
#   loss_value = maximum_mean_discrepancy(
#       source_samples, target_samples, kernel=gaussian_kernel)
#   loss_value = tf.maximum(1e-4, loss_value) * weight
#   assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
#   with tf.control_dependencies([assert_op]):
#     tag = 'MMD Loss'
#     if scope:
#       tag = scope + tag
#     tf.summary.scalar(tag, loss_value)
#     tf.losses.add_loss(loss_value)
#
#   return loss_value
