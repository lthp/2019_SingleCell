import sys
import os
import pandas as pd

sys.path.append('..')
sys.path.append('/Users/laurieprelot/Documents/Projects/2019_Deep_learning/git_DL')
datapath = '/Users/laurieprelot/Documents/Projects/2019_Deep_learning/data/Chevrier-et-al'
from loading_and_preprocessing.data_loader import load_data_basic
from baselines.baselines_helpers import scale, convert_to_ann, sample_cells, batch_correct, prep_anndata_for_eval, eval_batch_sample
from baselines.baselines_wrappers import wrapper_raw, wrapper_combat, wrapper_reg, wrapper_mnn




# global settings

path = os.path.join(datapath, 'chevrier_data_pooled_full_panels.parquet')
wd = '/Users/laurieprelot/Documents/Projects/2019_Deep_learning/data/Chevrier-et-al'
#data_path = os.path.join(wd,'/Dataset5/')
save_path = os.path.join(wd,'baselines')

# load data
### Load Module
df_full = None
for sample in ['sample5','sample75','sample65']:
    x1_train, x1_test, x2_train, x2_test =load_data_basic(path,
                                 sample=sample, batch_names=['batch1', 'batch3'], panel=None)


    tmp_  = pd.concat([x1_train, x2_train])
    if df_full is None:
        df_full = tmp_
    else:
        df_full = pd.concat([df_full, tmp_], axis = 0 )

metadata_batch = [ i.split('_')[0] for i in df_full.index]
metadata_cell = [ i.split('_')[-1] for i in df_full.index]
df_full['metadata_batch'] = metadata_batch
df_full['metadata_celltype'] = metadata_cell
df_full['metadata_sample'] = sample
df_full = df_full.dropna(axis=1)
df_full = df_full.reset_index(drop = True)





adata_full = convert_to_ann(df_full, sample_col_name = "metadata_sample", batch_col_name="metadata_batch",
                  celltype_col_name = 'metadata_celltype')
adata_full.obs_names_make_unique()

# for a quick run subset the
#data to 3 selected samples
samples_selected = ['sample5','sample75','sample65']
adata_full = adata_full[adata_full.obs['sample'].isin(samples_selected),:].copy()



eval_full_raw = wrapper_raw(adata_full, samples_selected, save_path)
eval_full_batch_reg = wrapper_reg(adata_full, samples_selected, save_path)
eval_full_batch_combat = wrapper_combat(adata_full, samples_selected, save_path)
eval_full_batch_mnn_mean = wrapper_mnn(adata_full, samples_selected, save_path)

# merge all baseline scores
eval_all = pd.concat([eval_full_raw, eval_full_batch_reg, eval_full_batch_combat, eval_full_batch_mnn_mean])
eval_all = eval_all.sort_values(by=['sample', 'method'])
eval_all.to_csv(save_path+'scores_baselines_full.csv')
eval_all

