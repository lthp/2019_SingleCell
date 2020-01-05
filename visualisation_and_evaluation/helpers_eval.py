import numpy as np
import pandas as pd
import os 
import glob
import pdb
import scipy as sp
from universal_divergence import estimate
from sklearn.metrics import silhouette_samples
from math import log, e
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import umap
import scipy as sp
from scipy.special import expit

### adapted from BERMUDA and ensured the random seeds are set where appropriate (= slightly modified)
def cal_UMAP(code, pca_dim = 50, n_neighbors = 30, min_dist=0.1, n_components=2, metric='cosine', random_state=0):
    """ Calculate UMAP dimensionality reduction
    Args:
        code: num_cells * num_features
        pca_dim: if dimensionality of code > pca_dim, apply PCA first
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        n_components: UMAP parameter
        metric: UMAP parameter
        random_state: random seed
    Returns:
        umap_code: num_cells * n_components
    """
    if code.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        code = pca.fit_transform(code)
    fit = umap.UMAP(n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=n_components,
                    metric=metric,
                    random_state=random_state)
    umap_code = fit.fit_transform(code)

    return umap_code

def entropy(labels, base=None):
    """ Computes entropy of label distribution.
    Args:
        labels: list of integers
    Returns:
        ent: entropy
    """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent


def cal_entropy(code, idx, dataset_labels, k=100):
    """ Calculate entropy of cell types of nearest neighbors
    Args:
        code: num_cells * num_features, embedding for calculating entropy
        idx: binary, index of observations to calculate entropy
        dataset_labels:
        k: number of nearest neighbors
    Returns:
        entropy_list: list of entropy of each cell
    """
    cell_sample = np.where(idx == True)[0]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(code)
    entropy_list = []
    _, indices = nbrs.kneighbors(code[cell_sample, :])
    for i in range(len(cell_sample)):
        entropy_list.append(entropy(dataset_labels[indices[i, :]]))

    return entropy_list


def evaluate_scores(div_ent_code, sil_code, cell_labels, dataset_labels, num_datasets,
                    div_ent_dim, sil_dim, sil_dist, random_state, cal_min=30):
    """ Calculate three proposed evaluation metrics
    Args:
        div_ent_code: num_cells * num_features, embedding for divergence and entropy calculation, usually with dim of 2
        sil_code: num_cells * num_features, embedding for silhouette score calculation
        cell_labels:
        dataset_labels:
        num_datasets:
        div_ent_dim: if dimension of div_ent_code > div_ent_dim, apply PCA first
        sil_dim: if dimension of sil_code > sil_dim, apply PCA first
        sil_dist: distance metric for silhouette score calculation
        random_state: random seed
        cal_min: minimum number of cells for estimation
    Returns:
        div_score: divergence score
        ent_score: entropy score
        sil_score: silhouette score
    """
    # calculate divergence and entropy
    if div_ent_code.shape[1] > div_ent_dim:
        div_ent_code = PCA(n_components=div_ent_dim, random_state=random_state).fit_transform(div_ent_code)
    div_pq = []  # divergence dataset p, q
    div_qp = []  # divergence dataset q, p
    ent = []  # entropy
    # pairs of datasets
    for d1 in range(1, num_datasets+1):
        for d2 in range(d1+1, num_datasets+1):
            idx1 = dataset_labels == d1
            idx2 = dataset_labels == d2
            labels = np.intersect1d(np.unique(cell_labels[idx1]), np.unique(cell_labels[idx2]))
            idx1_mutual = np.logical_and(idx1, np.isin(cell_labels, labels))
            idx2_mutual = np.logical_and(idx2, np.isin(cell_labels, labels))
            idx_specific = np.logical_and(np.logical_or(idx1, idx2), np.logical_not(np.isin(cell_labels, labels)))
            # divergence
            if np.sum(idx1_mutual) >= cal_min and np.sum(idx2_mutual) >= cal_min:
                div_pq.append(max(estimate(div_ent_code[idx1_mutual, :], div_ent_code[idx2_mutual, :], cal_min), 0))
                div_qp.append(max(estimate(div_ent_code[idx2_mutual, :], div_ent_code[idx1_mutual, :], cal_min), 0))
            # entropy
            if (sum(idx_specific) > 0):
                ent_tmp = cal_entropy(div_ent_code, idx_specific, dataset_labels)
                ent.append(sum(ent_tmp) / len(ent_tmp))
    if len(ent) == 0:  # if no dataset specific cell types, store entropy as -1
        ent.append(-1)

    # calculate silhouette_score (only if more than one cell-type provided)
    if(len(set(cell_labels))==1):
        sil_scores = [np.nan]
    else:  
        if sil_code.shape[1] > sil_dim:
            sil_code = PCA(n_components=sil_dim,random_state=random_state).fit_transform(sil_code)
        sil_scores = silhouette_samples(sil_code, cell_labels, metric=sil_dist)

    # average for scores
    div_score = (sum(div_pq) / len(div_pq) + sum(div_qp) / len(div_qp)) / 2
    ent_score = sum(ent) / len(ent)
    sil_score = sum(sil_scores) / len(sil_scores)

    return div_score, ent_score, sil_score

# helper functions to convert the model output into BERMUDA-required format for evaluation
def separate_metadata(data):
    """
    Function to create metadata from data index
    inputs:
        pandas dataframe with all metainfo in the index, separated by '_'
    outputs:
        data, metadata
    """
    df = data.copy()
    assert(df.index.nlevels==1)
    metainfo = df.index.to_frame(name='metainfo').metainfo.str.split("_", n = -1, expand = True) 
    metainfo.columns = ["".join(e for e in x if e.isalpha()) for x in metainfo.iloc[0,:]]
    if(len([x for x in metainfo.columns if 'celltype' in x])):
        metainfo.columns = ['cell_type' if 'celltype' in x else x for x in metainfo.columns]
        metainfo['cell_type'] = [x.split('celltype')[-1] for x in metainfo['cell_type']]
    else:
        metainfo['cell_type'] = 'all'
    df.index = [t+'_cell'+str(i) for (i,t) in enumerate(df.index)]
    metainfo.index = df.index
    return(df, metainfo)

def prep_data_for_eval(data, metadata, umap_dim=20, random_state=0):
    """
    Function to convert the data and metadata into a format required by the vealuation functions
    inputs:
        data: (batch-corrected) data containing all cells
        metadata: corresopnding metadata with index shared with data, column 'batch' and column 'cell_type'
        random_state: random seed
    outputs:
        umap_codes, data, cell_type_labels, batch_labels, number_of_datasets
    """
    assert(data.index.nlevels==1)
    idx = list(data.index)
    # get batch labels
    batch_labels = metadata.loc[idx,'batch']
    ct_labels = metadata.loc[idx,'cell_type']
    
    num_datasets = len(set(batch_labels))
    batch_dict = dict(zip(set(batch_labels), range(len(set(batch_labels)))))
    batch_labels_num = np.array([batch_dict[x]+1 for x in batch_labels])
    ct_dict = dict(zip(set(ct_labels), range(len(set(ct_labels)))))
    ct_labels_num = np.array([ct_dict[x]+1 for x in ct_labels])
    
    data = np.array(data)
    umap_codes = cal_UMAP(data, umap_dim, random_state=random_state)
    return(umap_codes, data, ct_labels_num, batch_labels_num, num_datasets)

def extract_scores(path_dir, fname):
    """
    Function to extract the evaluation scores and metainformation
    inputs:
        path_dir: path to the directory with files
        fname: file name
    outputs:
        pandas dataframe with scores and metainformation
    """
    df = pd.read_csv(os.path.join(path_dir,fname), header=None, index_col=[0])
    # structure of saved scores from DL models: index=epoch, columns=[divergence_score, entropy_score, silhouette_score]
    df.columns = ['divergence_score', 'entropy_score', 'silhouette_score']
    df.index_name = ['epoch']
    df['silhouette_score_neg'] = -df.loc[:,'silhouette_score']
    df['method'] = fname.split('scores_')[-1].split('_sample')[0].replace('_',' ')
    df['sample'] = fname.split('_')[-1].split('.csv')[0]
    return(df)

def wa(scores, weights=[0.5,0.3,0.2]):
    """
    Function to calculate a weighted average of the scores after trafo via sigmoid function
    inputs:
        vectores with 3 values corresponding to ['divergence_score', 'entropy_score', 'silhouette_score_neg']
    outputs:
        weighted average
    """
    # map all scores to (0,1) via sigmoid function
    scores = expit(scores)
    divergence_score, entropy_score, silhouette_score_neg = scores
    wa = weights[0]*divergence_score + weights[1]*entropy_score + weights[2]*silhouette_score_neg
    return(wa)


def select_best_run(df, method='div'):
    """
    Function to select the best run wrt scores
    inputs:
        df: pd dataframe with scores and metainformation
        method: method to select the scores {'div': based on divergence score only, 
                                             'wa': weighted average of the 3 scores}
    outputs: 
        pd dataframe of 1 row with the scores and metainformation for the best run
    """
    if(method=='div'):
        df_best = df.loc[[df['divergence_score'].idxmin()],:]
    elif(method=='wa'):
        wa_score = df[['divergence_score', 'entropy_score', 'silhouette_score_neg']].apply(lambda x: wa(x), axis=1)
        df_best = df.loc[[wa_score.idxmin()],:]
    else:
        print('please provide a valid selection method')
    return(df_best)