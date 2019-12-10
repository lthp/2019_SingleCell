import fcsparser
import numpy as np
from numpy import genfromtxt
from auto_encoder_updated import file_io as io
import os.path
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split


class Sample:
    x = None
    y = None

    def __init__(self, x, y=None):
        self.x = x
        self.y = y


def preprocess_samples(sample):
    sample.x = np.log(1 + np.abs(sample.x))
    return sample


def standard_scale(sample, preprocessor=None):
    if preprocessor is None:
        preprocessor = prep.StandardScaler().fit(sample.x)
    sample.x = preprocessor.transform(sample.x)
    return sample, preprocessor


def load_data(data_path, data_index, relevant_markers, mode, skip_header=0):
    if mode == 'CSV':
        data_filename = data_path + '/sample' + str(data_index) + '.csv'
        x = genfromtxt(os.path.join(io.deep_learning_root(), data_filename), delimiter=',', skip_header=skip_header)
    if mode == 'FCS':
        files = [file for file in os.listdir(data_path) if '.fcs' in file]
        data_filename = os.path.join(data_path, files[data_index])
        _, x = fcsparser.parse(os.path.join(io.deep_learning_root(), data_filename), reformat_meta=True)
        x = x.as_matrix()
    x = x[:, relevant_markers]
    #label_filename = data_path + '/labels' + str(data_index) + '.csv'
    #labels = genfromtxt(os.path.join(io.deep_learning_root(), label_filename), delimiter=',')
    #labels = np.int_(labels)
    sample = Sample(x)

    return sample

def load_data_tsv(data_path, data_index, relevant_markers, skip_header=0):
    print('Loading data from .tsv')
    files = [os.path.join(data_path, file) for file in os.listdir(data_path) if '.tsv' in file]
    samples = []
    for i in data_index:
        x = genfromtxt(os.path.join(io.deep_learning_root(), files[i]), delimiter='\t', skip_header=skip_header)
        x = x[:, relevant_markers]
        sample = Sample(x)
        samples.append(sample)
    return samples

def split_data(sample, test_size):
    data_train, data_test, label_train, label_test = train_test_split(sample.x, sample.y, test_size=test_size)

    train_sample = Sample(data_train, label_train)
    test_sample = Sample(data_test, label_test)
    return train_sample, test_sample


def choose_reference_sample(data_path, relevant_markers, mode):
    samples = load_data(data_path, relevant_markers, mode)
    samples = [preprocess_samples(sample) for sample in samples]

    num_samples = len(samples)
    norms = np.zeros(shape=[num_samples, num_samples])
    for i in range(num_samples):
        cov_i = np.cov(samples[i].x, rowvar=False)
        for j in range(num_samples):
            cov_j = np.cov(samples[j].x, rowvar=False)
            cov_diff = cov_i - cov_j
            norms[i, j] = np.linalg.norm(cov_diff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_sample_ind = np.argmin(avg)
    return ref_sample_ind
