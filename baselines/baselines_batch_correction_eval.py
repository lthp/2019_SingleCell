# Script to compute evaluation scores for raw data as well as batch corrected using besline methods: regressing batch effect out, ComBat and mnnCorrect. The workflow is run for both, the simulated and real-world data.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os 
import glob
import sys
import pdb
from sklearn import preprocessing
import scipy as sp
import matplotlib as mpl
mpl.use('TkAgg') 
import anndata
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.getcwd()))
from visualisation_and_evaluation.helpers_eval import cal_UMAP, entropy, cal_entropy, evaluate_scores, separate_metadata
from baselines.baselines_helpers import scale, convert_to_ann, sample_cells, batch_correct, prep_anndata_for_eval, eval_batch_sample, load_sample_data
from baselines.baselines_wrappers import wrapper_raw, wrapper_reg, wrapper_combat, wrapper_mnn

# ####################  simulated data  ###################
# ######  all cell populations shared  ######
# data_path = '../data/toy_data_gamma_w_index.parquet'
# save_path = '../eval_scores/'
# df_full = load_sample_data(data_path, samples_selected=['sample1'], batch_names= ['batch1', 'batch2'])
# samples_selected = sp.unique(df_full['metadata_sample'])
# adata_full = convert_to_ann(df_full, sample_col_name = "metadata_sample", batch_col_name="metadata_batch",
#                   celltype_col_name = 'metadata_celltype')
# adata_full.obs_names_make_unique()
# suffix = 'toy'
# eval_full_raw = wrapper_raw(adata_full, samples_selected, save_path, suffix)
# eval_full_batch_reg = wrapper_reg(adata_full, samples_selected, save_path, suffix)
# eval_full_batch_combat = wrapper_combat(adata_full, samples_selected, save_path, suffix)
# eval_full_batch_mnn_mean = wrapper_mnn(adata_full, samples_selected, save_path, suffix)

# # merge all baseline scores
# eval_all = pd.concat([eval_full_raw, eval_full_batch_reg, eval_full_batch_combat, eval_full_batch_mnn_mean])
# eval_all.to_csv(save_path+'scores_baselines_'+suffix+'_upsample.csv')
# print("simulated data with all cell populations shared analysed successfully")

# ######  some cell populations shared  ######
# data_path = '../data/toy_data_gamma_w_index_subset.parquet'
# save_path = '../eval_scores/'
# df_full = load_sample_data(path, samples_selected=['sample1'], batch_names= ['batch1', 'batch2'])
# samples_selected = sp.unique(df_full['metadata_sample'])
# adata_full = convert_to_ann(df_full, sample_col_name = "metadata_sample", batch_col_name="metadata_batch",
#                   celltype_col_name = 'metadata_celltype')
# adata_full.obs_names_make_unique()
# suffix = 'toysubset'
# eval_full_raw = wrapper_raw(adata_full, samples_selected, save_path, suffix)
# eval_full_batch_reg = wrapper_reg(adata_full, samples_selected, save_path, suffix)
# eval_full_batch_combat = wrapper_combat(adata_full, samples_selected, save_path, suffix)
# eval_full_batch_mnn_mean = wrapper_mnn(adata_full, samples_selected, save_path, suffix)

# # merge all baseline scores
# eval_all = pd.concat([eval_full_raw, eval_full_batch_reg, eval_full_batch_combat, eval_full_batch_mnn_mean])
# eval_all.to_csv(save_path+'scores_baselines_'+suffix+'_upsample.csv')
# print("simulated data with only some cell populations shared analysed successfully")

####################  Chevrier data  ###################
data_path = '../data/chevrier_samples_5_65_75.parquet'
save_path = '../eval_scores/'

df_full = load_sample_data(data_path, samples_selected=['sample5','sample75','sample65'], batch_names= ['batch1', 'batch3'])
adata_full = convert_to_ann(df_full, sample_col_name = "metadata_sample", batch_col_name="metadata_batch",
                  celltype_col_name = 'metadata_celltype')
adata_full.obs_names_make_unique()
# for a quick run subset the data to 3 selected samples
samples_selected = ['sample5','sample75','sample65']
adata_full = adata_full[adata_full.obs['sample'].isin(samples_selected),:].copy()

suffix='full'
eval_full_raw = wrapper_raw(adata_full, samples_selected, save_path, suffix)
eval_full_batch_reg = wrapper_reg(adata_full, samples_selected, save_path, suffix)
eval_full_batch_combat = wrapper_combat(adata_full, samples_selected, save_path, suffix)
eval_full_batch_mnn_mean = wrapper_mnn(adata_full, samples_selected, save_path, suffix)

# merge all baseline scores
eval_all = pd.concat([eval_full_raw, eval_full_batch_reg, eval_full_batch_combat, eval_full_batch_mnn_mean])
eval_all = eval_all.sort_values(by=['sample', 'method'])
eval_all.to_csv(save_path+'scores_baselines_full_upsample.csv')
print("real-world data with analysed successfully")