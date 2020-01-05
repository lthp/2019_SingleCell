
# coding: utf-8

# ### README
# A notebook to compute evaluation scores for raw data as well as bacth corrected using besline methods: regressing batch effect out, ComBat and mnnCorrect. The workflow is run for both, the simulated and real-world data.

# In[2]:


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


# In[ ]:


def wrapper_raw(adata_full, samples_selected, save_path, suffix='full'):
    # raw scores
    adata_full_dict = dict()
    for sample in samples_selected:
        adata_full_dict[sample] = adata_full[adata_full.obs['sample']==sample].copy()
    eval_full_raw = eval_batch_sample(adata_full_dict)
    eval_full_raw['method'] = 'raw'
    eval_full_raw.to_csv(save_path+'scores_raw_'+suffix+'.csv')
    return(eval_full_raw)

def wrapper_reg(adata_full, samples_selected, save_path):
    # regress out batch effect
    adata_full_batch_reg = batch_correct(adata_full, method='reg')
    eval_full_batch_reg = eval_batch_sample(adata_full_batch_reg)
    eval_full_batch_reg['method'] = 'reg'
    eval_full_batch_reg.to_csv(save_path+'scores_reg_'+suffix+'.csv')
    return(eval_full_batch_reg)

def wrapper_combat(adata_full, samples_selected, save_path):
    # combat
    adata_full_batch_combat = batch_correct(adata_full, method='combat')
    eval_full_batch_combat = eval_batch_sample(adata_full_batch_combat)
    eval_full_batch_combat['method'] = 'combat'
    eval_full_batch_combat.to_csv(save_path+'scores_combat_'+suffix+'.csv')
    return(eval_full_batch_combat)

def wrapper_mnn(adata_full, samples_selected, save_path):
    # mnnCorrect
    adata_batch_mnn = dict()
    max_cells = 1000
    random_state_list = [123465, 87654, 289, 243, 1234]
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


# In[10]:


wd = os.getcwd()
wd = os.path.abspath(os.path.join(wd,"..","..","data"))


# In[ ]:


####################  simulated data  ###################


# In[ ]:


######  all cell populations shared  ######


# In[ ]:


# global settings
data_path = os.path.join(wd,'/simulated/')
save_path = os.path.join(wd,'/simulated/eval_scores/')

df_full = pd.read_parquet(data_path+'toy_data_gamma_w_index.parquet')
df_full = df_full.dropna(axis=1)
samples_selected = sp.unique(df_full['metadata_sample'])
adata_full = convert_to_ann(df_full, sample_col_name = "metadata_sample", batch_col_name="metadata_batch",
                  celltype_col_name = 'metadata_celltype')
adata_full.obs_names_make_unique()


# In[ ]:


suffix = 'toy'
eval_full_raw = wrapper_raw(adata_full, samples_selected, save_path, suffix)
eval_full_batch_reg = wrapper_reg(adata_full, samples_selected, save_path, suffix)
eval_full_batch_combat = wrapper_combat(adata_full, samples_selected, save_path, suffix)
eval_full_batch_mnn_mean = wrapper_mnn(adata_full, samples_selected, save_path, suffix)

# merge all baseline scores
eval_all = pd.concat([eval_full_raw, eval_full_batch_reg, eval_full_batch_combat, eval_full_batch_mnn_mean])
eval_all.to_csv(save_path+'scores_baselines_full.csv')
eval_all


# In[ ]:


######  some cell populations shared  ######


# In[ ]:


# global settings
data_path = os.path.join(wd,'/simulated/')
save_path = os.path.join(wd,'/simulated/eval_scores_subset/')

df_full = pd.read_parquet(data_path+'toy_data_gamma_w_index_subset.parquet')
df_full = df_full.dropna(axis=1)
samples_selected = sp.unique(df_full['metadata_sample'])
adata_full = convert_to_ann(df_full, sample_col_name = "metadata_sample", batch_col_name="metadata_batch",
                  celltype_col_name = 'metadata_celltype')
adata_full.obs_names_make_unique()


# In[ ]:


suffix = 'toysubset'
eval_full_raw = wrapper_raw(adata_full, samples_selected, save_path, suffix)
eval_full_batch_reg = wrapper_reg(adata_full, samples_selected, save_path, suffix)
eval_full_batch_combat = wrapper_combat(adata_full, samples_selected, save_path, suffix)
eval_full_batch_mnn_mean = wrapper_mnn(adata_full, samples_selected, save_path, suffix)

# merge all baseline scores
eval_all = pd.concat([eval_full_raw, eval_full_batch_reg, eval_full_batch_combat, eval_full_batch_mnn_mean])
eval_all.to_csv(save_path+'scores_baselines_full.csv')
eval_all


# In[ ]:


####################  Chevrier data  ###################


# In[ ]:


# global settings
data_path = os.path.join(wd,'/Dataset5/')
save_path = os.path.join(wd,'/Dataset5/eval_scores/')

# load data
df_full = pd.read_parquet(data_path+'chevrier_data_pooled_full_panels.parquet')
df_full = df_full.dropna(axis=1)
adata_full = convert_to_ann(df_full, sample_col_name = "metadata_sample", batch_col_name="metadata_batch",
                  celltype_col_name = 'metadata_celltype')
adata_full.obs_names_make_unique()
# for a quick run subset the 
data to 3 selected samples
samples_selected = ['sample5','sample75','sample65']
adata_full = adata_full[adata_full.obs['sample'].isin(samples_selected),:].copy()


# In[ ]:


eval_full_raw = wrapper_raw(adata_full, samples_selected, save_path)
eval_full_batch_reg = wrapper_reg(adata_full, samples_selected, save_path)
eval_full_batch_combat = wrapper_combat(adata_full, samples_selected, save_path)
eval_full_batch_mnn_mean = wrapper_mnn(adata_full, samples_selected, save_path)

# merge all baseline scores
eval_all = pd.concat([eval_full_raw, eval_full_batch_reg, eval_full_batch_combat, eval_full_batch_mnn_mean])
eval_all = eval_all.sort_values(by=['sample', 'method'])
eval_all.to_csv(save_path+'scores_baselines_full.csv')
eval_all

