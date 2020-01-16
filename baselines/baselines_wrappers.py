import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os 
import glob
import sys
from FlowCytometryTools import FCMeasurement
from collections import Counter
import pdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import xlrd
from collections import Counter
from sklearn import preprocessing
import scipy as sp
import anndata
import scanpy as sc

sys.path.append(os.path.dirname(os.getcwd()))
from visualisation_and_evaluation.helpers_eval import cal_UMAP, entropy, cal_entropy, evaluate_scores, separate_metadata
from baselines.baselines_helpers import scale, convert_to_ann, sample_cells, batch_correct, prep_anndata_for_eval, eval_batch_sample 


def wrapper_raw(adata_full, samples_selected, save_path, suffix='full'):
    # raw scores
    adata_full_dict = dict()
    for sample in samples_selected:
        adata_full_dict[sample] = adata_full[adata_full.obs['sample']==sample].copy()
    eval_full_raw = eval_batch_sample(adata_full_dict)
    eval_full_raw['method'] = 'raw'
    eval_full_raw.to_csv(save_path+'scores_raw_'+suffix+'.csv')
    return(eval_full_raw)

def wrapper_reg(adata_full, samples_selected, save_path, suffix='full'):
    # regress out batch effect
    adata_full_batch_reg = batch_correct(adata_full, method='reg')
    eval_full_batch_reg = eval_batch_sample(adata_full_batch_reg)
    eval_full_batch_reg['method'] = 'reg'
    eval_full_batch_reg.to_csv(save_path+'scores_reg_'+suffix+'.csv')
    return(eval_full_batch_reg)

def wrapper_combat(adata_full, samples_selected, save_path, suffix='full'):
    # combat
    adata_full_batch_combat = batch_correct(adata_full, method='combat')
    eval_full_batch_combat = eval_batch_sample(adata_full_batch_combat)
    eval_full_batch_combat['method'] = 'combat'
    eval_full_batch_combat.to_csv(save_path+'scores_combat_'+suffix+'.csv')
    return(eval_full_batch_combat)

def wrapper_mnn(adata_full, samples_selected, save_path, suffix='full'):
    # mnnCorrect
    adata_batch_mnn = dict()
    max_cells = 1000
    #random_state_list = [123465, 87654, 289, 243, 1234]
    random_state_list = [19885, 1998, 8768, 26998, 243]
    eval_random_state = dict()
    for random_state in random_state_list:
        for sample in samples_selected:
            adata = adata_full[adata_full.obs['sample']==sample,:].copy()
            adata_sampled = sample_cells(adata, random_state=random_state, max_cells=max_cells)
            adata_sampled_batch_ann = batch_correct(adata_sampled, method='mnn')
            adata_batch_mnn[sample] = adata_sampled_batch_ann[sample]
        eval_full_mnn = eval_batch_sample(adata_batch_mnn)
        eval_random_state[random_state] = eval_full_mnn
    eval_full_batch_mnn = pd.concat(eval_random_state)
    eval_full_batch_mnn['random_state'] = [x for x in eval_full_batch_mnn.index.get_level_values(0)]
    eval_full_batch_mnn.index = range(eval_full_batch_mnn.shape[0])
    eval_full_batch_mnn.to_csv(save_path+'scores_mnn_'+suffix+'.csv')
    # average score scross random_states
    eval_full_batch_mnn['divergence_score'] = pd.to_numeric(eval_full_batch_mnn['divergence_score'])
    eval_full_batch_mnn['entropy_score'] = pd.to_numeric(eval_full_batch_mnn['entropy_score'])
    eval_full_batch_mnn['silhouette_score'] = pd.to_numeric(eval_full_batch_mnn['silhouette_score'])
    eval_full_batch_mnn = eval_full_batch_mnn.drop(columns='random_state')
    eval_full_batch_mnn.groupby(['sample']).apply(np.mean)
    eval_full_batch_mnn_mean = pd.DataFrame(eval_full_batch_mnn.groupby(['sample']).apply(np.mean))
    eval_full_batch_mnn_mean['method'] = 'mnn'
    eval_full_batch_mnn_mean['sample'] = eval_full_batch_mnn_mean.index
    eval_full_batch_mnn_mean.to_csv(save_path+'scores_mnn_'+suffix+'_mean.csv')
    return(eval_full_batch_mnn_mean)

