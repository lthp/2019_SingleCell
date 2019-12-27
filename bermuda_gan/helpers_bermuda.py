# !/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale
from imblearn.over_sampling import RandomOverSampler
imblearn_seed = 0
np.random.seed(12345)
import tensorflow as tf

def loader(dataset_file_list, take_log, oversample, standardization, scaling):
    """ Read TPM data of a dataset saved in csv format
    Format of the csv:
    first row: sample labels
    second row: cell labels
    third row: cluster labels from Seurat
    first column: gene symbols
    Args:
        filename: name of the csv file
        take_log: whether do log-transformation on input data
        standardize: whether standardization
        scaling: whather to scale between zero and one
    Returns:
        dataset: a dict with keys 'gene_exp', 'gene_sym', 'sample_labels', 'cell_labels', 'cluster_labels'
    """
    all_classes = np.array([])
    classes_per_dataset = []
    for filename in dataset_file_list:
        df = pd.read_csv(filename, header=None, nrows = 3)
        dat = df[df.columns[1:]].values
        cluster_labels = dat[2, :].astype(int)
        classes_per_dataset.append(len(np.unique(cluster_labels)))
        all_classes = np.concatenate([ all_classes, cluster_labels ])
    major_class = max(np.unique(all_classes, return_counts = True)[1])
    resampling_size = np.cumproduct(classes_per_dataset)[-1] * \
                      np.int((major_class / np.min(classes_per_dataset) + 1))
    # Resampling heuristics: the number of sample in the resampled datasets
    # should be a multiple of the number of the number of classes in each dataset and
    # and the number of sample per class be >= to the number of sample in the major class



    dataset_list = []
    for filename in dataset_file_list:
        dataset = {}
        df = pd.read_csv(filename, header=None)
        dat = df[df.columns[1:]].values
        sample_labels = dat[0, :].astype(int)
        cell_labels  =  dat[1, :].astype(int)
        cluster_labels = dat[2, :].astype(int)
        gene_sym = df[df.columns[0]].tolist()[3:]
        dataset['gene_sym'] = gene_sym
        ### Gene expression
        gene_exp = dat[3:, :]
        if take_log:
            gene_exp = np.log2(gene_exp + 1)
        if standardization:
            scale(gene_exp, axis=1, with_mean=True, with_std=True, copy=False)
        if scaling:  # scale to [0,1]
            minmax_scale(gene_exp, feature_range=(0, 1), axis=1, copy=False)
        if oversample:
            target_sizes = {}
            cluster_q = np.unique(cluster_labels)
            for class_ in cluster_q:
                target_sizes[class_] = int  (resampling_size / len(cluster_q))
            gene_exp = gene_exp.transpose()
            gene_exp, cluster_labels, sampling_idx = RandomOverSampler(random_state=imblearn_seed, return_indices = True, sampling_strategy = target_sizes ).fit_sample(gene_exp, cluster_labels )
            cell_labels = cell_labels[sampling_idx]
            sample_labels = sample_labels[sampling_idx]
            gene_exp = gene_exp.transpose()

        dataset['sample_labels'] = sample_labels #TODO remove
        dataset['cell_labels'] = cell_labels
        dataset['gene_exp'] = gene_exp
        dataset['cluster_labels'] = cluster_labels

        dataset_list.append(dataset)
    return dataset_list



def  train_test(dataset, split = 0.80):
    ''' splits the dataset between train and test set. ASSUMES ONLY TWO BATCHES
      Args:
          split: percentage of the training set
      Returns:
          x1_train, x1_test, x2_train, x2_test : dictionaries which are dict with keys 'gene_exp', 'gene_sym', 'sample_labels', 'cell_labels', 'cluster_labels'
    '''
    x1_train = {}
    x1_test ={}
    x2_train = {}
    x2_test = {}
    x1 = dataset[0]
    x2 = dataset[1]
    n1 = x1['gene_exp'].shape[1]
    n2 = x2['gene_exp'].shape[1]
    idx1 = np.random.permutation(n1)
    idx2 = np.random.permutation(n2)
    c1 = int(np.ceil(split * n1 ))
    c2 = int(np.ceil(split * n2 ))
    for name in ['sample_labels', 'cell_labels', 'cluster_labels', 'gene_exp', 'gene_sym']:
        if name == 'gene_exp':
            x1_train[name] = x1[name][:, idx1[:c1]]
            x1_test[name] = x1[name][:, idx1[c1:]]
            x2_train[name] = x2[name][:, idx2[:c2]]
            x2_test[name] = x2[name][:, idx2[c2:]]
        elif name == 'gene_sym':
            x1_train[name] = x1[name]
            x1_test[name] = x1[name]
            x2_train[name] = x2[name]
            x2_test[name] = x2[name]
        else:
            x1_train[name] = x1[name][idx1[:c1]]
            x1_test[name] = x1[name][idx1[c1:]]
            x2_train[name] = x2[name][idx2[:c2]]
            x2_test[name] = x2[name][idx2[c2:]]

    return x1_train, x1_test, x2_train, x2_test

def read_cluster_similarity(filename, thr):
    """ read cluster similarity matrix, convert into the format of pairs and weights
    first line is cluster label, starting with 1
    Args:
        filename: filename of the cluster similarity matrix
        thr: threshold for identifying corresponding clusters
    Returns:
        cluster_pairs: np matrix, num_pairs by 3 matrix
                        [cluster_idx_1, cluster_id_2, weight]
    """
    df = pd.read_csv(filename, header=None)
    cluster_matrix = df[df.columns[:]].values
    cluster_matrix = cluster_matrix[1:, :]
    # use blocks of zeros to determine which clusters belongs to which datasets
    num_cls = cluster_matrix.shape[0]
    dataset_idx = np.zeros(num_cls, dtype=int)
    idx = 0
    for i in range(num_cls - 1):
        dataset_idx[i] = idx
        if cluster_matrix[i, i + 1] != 0:
            idx += 1
    dataset_idx[num_cls - 1] = idx
    num_datasets = idx + 1

    # only retain pairs if cluster i in dataset a is most similar to cluster j in dataset b
    local_max = np.zeros(cluster_matrix.shape, dtype=int)
    for i in range(num_cls):
        for j in range(num_datasets):
            if dataset_idx[i] == j:
                continue
            tmp = cluster_matrix[i, :] * (dataset_idx == j)
            local_max[i, np.argmax(tmp)] = 1
    local_max = local_max + local_max.T
    local_max[local_max > 0] = 1
    cluster_matrix = cluster_matrix * local_max  # only retain dataset local maximal pairs
    cluster_matrix[cluster_matrix < thr] = 0
    cluster_matrix[cluster_matrix > 0] = 1 # binarize

    # construct cluster pairs
    tmp_idx = np.nonzero(cluster_matrix)
    valid_idx = tmp_idx[0] < tmp_idx[1]  # remove duplicate pairs
    cluster_pairs = np.zeros((sum(valid_idx), 3), dtype=float)
    cluster_pairs[:, 0] = tmp_idx[0][valid_idx] + 1
    cluster_pairs[:, 1] = tmp_idx[1][valid_idx] + 1
    for i in range(cluster_pairs.shape[0]):
        cluster_pairs[i, 2] = cluster_matrix[int(cluster_pairs[i, 0] - 1), int(cluster_pairs[i, 1] - 1)]

    return cluster_pairs


def remove_duplicate_genes(gene_exp, gene_sym):
    """ Remove duplicate gene symbols in a dataset
    Chooses the one with highest mean value when there are duplicate genes
    Args:
        gene_exp: np matrix, num_genes by num_cells
        gene_sym: length num_cells
    Returns:
        gene_exp
        gene_sym
    """
    dic = {}  # create a dictionary of gene_sym to identify duplicate
    for i in range(len(gene_sym)):
        if not gene_sym[i] in dic:
            dic[gene_sym[i]] = [i]
        else:
            dic[gene_sym[i]].append(i)
    if (len(dic) == len(gene_sym)):  # no duplicate
        return gene_exp, gene_sym

    remove_idx = []  # idx of gene symbols that will be removed
    for sym, idx in dic.items():
        if len(idx) > 1:  # duplicate exists
            # print('duplicate! ' + sym)
            remain_idx = idx[np.argmax(np.mean(gene_exp[idx, :], axis=1))]
            for i in idx:
                if i != remain_idx:
                    remove_idx.append(i)
    gene_exp = np.delete(gene_exp, remove_idx, 0)
    for idx in sorted(remove_idx, reverse=True):
        del gene_sym[idx]
    # print("Remove duplicate genes, remaining genes: {}".format(len(gene_sym)))

    return gene_exp, gene_sym


def intersection_idx(lists):
    """ intersection of multiple lists. Returns intersection and corresponding indexes
    Args:
        lists: list of lists that need to intersect
    Returns:
        intersect_list: list of intersection result
    """
    idx_dict_list = []  # create index dictionary
    for l in lists:
        idx_dict_list.append(dict((k, i) for i, k in enumerate(l)))
    intersect_set = set(lists[0])  # create intersection result
    for i in range(1, len(lists)):
        intersect_set = set(lists[i]).intersection(intersect_set)
    intersect_list = list(intersect_set)
    # generate corresponding index of intersection
    idx_list = []
    for d in idx_dict_list:
        idx_list.append([d[x] for x in intersect_list])

    return intersect_list, idx_list


def intersect_dataset(dataset_list):
    """ Only retain the intersection of genes among multiple datasets
    Args:
        dataset_list: list of datasets
    Returns:
        intersect_dataset_list: list of after intersection pf gene symbols
    """
    dataset_labels = ['sample_labels', 'cell_labels', 'cluster_labels']  # labels in a dataset
    intersect_dataset_list = []
    gene_sym_lists = []
    for i, dataset in enumerate(dataset_list):
        gene_sym_lists.append(dataset['gene_sym'])
    # intersection of gene symbols
    gene_sym, idx_list = intersection_idx(gene_sym_lists)
    # print("Intersection of genes: {}".format(len(gene_sym)))
    # only retain the intersection of genes in each dataset
    for dataset, idx in zip(dataset_list, idx_list):
        dataset_tmp = {'gene_exp': dataset['gene_exp'][idx,:], 'gene_sym': gene_sym}
        for l in dataset_labels:
            if l in dataset:
                dataset_tmp[l] = dataset[l]
        intersect_dataset_list.append(dataset_tmp)

    return intersect_dataset_list



def pre_processing(dataset_file_list, pre_process_paras):
    """ pre-processing of multiple datasets
    Args:
        dataset_file_list: list of filenames of datasets
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of datasets
    """
    # parameters
    take_log = pre_process_paras['take_log']
    standardization = pre_process_paras['standardization']
    scaling = pre_process_paras['scaling']
    oversample = pre_process_paras['oversample']
    split = pre_process_paras['split']

    dataset_list =  loader(dataset_file_list, take_log, oversample, standardization, scaling)
    dataset_list = intersect_dataset(dataset_list)  # retain intersection of gene symbols
    x1_train, x1_test, x2_train, x2_test = train_test(dataset_list, split)
    return x1_train, x1_test, x2_train, x2_test


# def make_mask(to_mask, positive_indices, sample_size):
#     """Args:
#     to_mask: tensor that will be masked [n_samples , features]
#     positive_indices: a tensor with i sample indices for which the rows of the mask should be True
#     Returns:
#         A boolean tensor with i rows True and n-i rows false
#     """
#     mask = None
#     for i in np.arange(sample_size):
#         if i in positive_indices:
#             extend = tf.expand_dims(tf.constant(np.ones(shape=sample_size)), 1)
#         else:
#             extend = tf.expand_dims(tf.constant(np.zeros(shape=sample_size)), 1)
#         if mask is not None:
#             mask = tf.concat([mask, extend], axis=1)
#         else:
#             mask = extend
#     mask = tf.transpose(mask)
#     return mask

def make_mask_np(to_mask, positive_indices):
    """Args:
    to_mask: tensor that will be masked [n_samples , features]
    positive_indices: a tensor with i sample indices for which the rows of the mask should be True
    Returns:
        A boolean tensor with i rows True and n-i rows false
    """
    mask = None
    for i in np.arange(to_mask.shape[0]):
        if i in positive_indices:
            extend = np.ones(shape=(to_mask.shape[0], 1) )
        else:
            extend = np.zeros(shape=(to_mask.shape[0], 1) )
        if mask is not None:
            mask = np.concatenate([mask, extend], axis=1)
        else:
            mask = extend

    return mask

def make_mask_tensor(x1, x2, x1_labels, x2_labels):
    classes_ = len(np.unique(x1_labels)) + len(np.unique(x2_labels)) + 1
    print(classes_)
    mask_tensor = np.empty(shape = (x1.shape[0],x1.shape[0],classes_))
    for j in np.unique(x1_labels):
        extract_cluster1 = np.where(x1_labels == j)[0]
        mask_tensor[:, : , j] = make_mask_np(x1, extract_cluster1)
    for j in np.unique(x2_labels):
        extract_cluster2 = np.where(x2_labels == j)[0]
        mask_tensor[:, :, j] = make_mask_np(x2, extract_cluster2)
    return mask_tensor





if __name__ == '__main__':
    dataset_file_list = ['data/muraro_seurat.csv', 'data/baron_human_seurat.csv']
    pre_process_paras = {'take_log': True, 'standardization': True, 'scaling': True, 'oversample': True}
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    print()