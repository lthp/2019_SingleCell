import numpy as np
from auto_encoder_updated import denoising_autoencoder as dae, data_hander as dh

# this info will change for each dataset
data_path = r'C:/Users/Public/PycharmProjects/deep/2019_DL_Class/auto_encoder_updated/Data/MultiCenter_16sample'
data_index = np.arange(1, 16+1)
train_index = data_index
test_index = data_index
relevant_markers = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])-1
mode = 'CSV'  # CSV or FCS (will be handled differently when loading the data)
num_classes = 4
keep_prob = .8

# Choose reference sample.
print('choose ref sample')
ref_sample_ind = dh.choose_reference_sample(data_path, train_index, relevant_markers, mode)
print('Load the target ' + str(train_index[ref_sample_ind]))
target = dh.load_data(data_path, train_index[ref_sample_ind], relevant_markers, mode)
# Pre-process sample.
target = dh.preprocess_samples(target)

# train denoising auto-encoder
print('Train the de-noising auto encoder.')
denoise = True
load = False
dataset_name = 'MultiCenter_16sample'
DAE = dae.trainDAE(target, data_path, ref_sample_ind, train_index, relevant_markers, mode, keep_prob, denoise, load,
                   dataset_name)
denoiseTarget = dae.predictDAE(target, DAE, denoise)
print('Done training the de-noising auto encoder.')

