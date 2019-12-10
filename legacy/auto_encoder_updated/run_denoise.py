import numpy as np
from auto_encoder_updated import denoising_autoencoder as dae, data_hander as dh, batch_effect_removal as ber

# this info will change for each dataset
data_path = r'C:\Users\heida\Documents\dl_data\Data_pooled'
relevant_markers = np.arange(30, 40)
mode = '.tsv'  # CSV or FCS (will be handled differently when loading the data)
#num_classes = 4
keep_prob = .8
train_index = [0]

# Choose reference sample.
# print('choose ref sample')
#ref_sample_ind = dh.choose_reference_sample(data_path, relevant_markers, mode)
# print('Load the target ' + str(train_index[ref_sample_ind]))
# target = dh.load_data(data_path, train_index[ref_sample_ind], relevant_markers, mode)
# # Pre-process sample.
# target = dh.preprocess_samples(target)

# train denoising auto-encoder
print('Train the de-noising auto encoder.')
denoise = True
load = False
dataset_name = 'MultiCenter_16sample'
DAE = dae.trainDAE(data_path, train_index, relevant_markers, mode, keep_prob, denoise, load,
                   dataset_name)
#denoiseTarget = dae.predictDAE(target, DAE, denoise)
print('Done training the de-noising auto encoder.')

