import numpy as np
import pandas as pd
import os 
import sys
import scipy as sp
import anndata
import scanpy as sc

sys.path.append(os.path.dirname(os.getcwd()))
from visualisation_and_evaluation.helpers_eval import cal_UMAP, entropy, cal_entropy, evaluate_scores, separate_metadata
from loading_and_preprocessing.data_loader import load_data_basic

def scale(x):
    p99 = np.percentile(x,99)
    x[x>p99] = p99
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return(x)


def convert_to_ann(data, sample_col_name = "metadata_sample", batch_col_name="metadata_batch",
                  celltype_col_name = 'metadata_celltype'):
    """
    Function to convert a pandas dataframe into anndata ( a common format for single-cell objects)
    inputs:
        data: pandas dataframe with expression and 'metadata' information
        sample_col_name: column name including sample IDs
        batch_col_name: column name including batch IDs
        celltype_col_name: column name including cell-types
    outputs:
        an anndata object with all column information in .var and row information in .obs
    """
    index_nlevels = data.index.nlevels
    row_attr = pd.DataFrame({'sample': data[sample_col_name], 'batch': data[batch_col_name],
                            'cell_type':data[celltype_col_name]})
    row_attr.index = data.index.get_level_values(index_nlevels-1)
    data = data.loc[:,~data.columns.str.startswith('metadata')]
    col_attr = pd.DataFrame({'marker_name': data.columns})
    col_attr.index = data.columns
    data_ann = anndata.AnnData(X=np.array(data),
                                obs=row_attr,
                                var=col_attr)
    #data_ann = data_ann.obs_names_make_unique()
    return(data_ann)

def sample_cells(adata, random_state=123465, max_cells=1000):
    """
    Function to sample cells from the batches since mnnCorrect cannot deal with such a high number of cells
    inputs:
        adata: anndata object containing data of two batches
        random_state: random seed for sampling
        max_cells: maximum number of cells to be samples from each batch
    outputs:
        anndata object containing data only for the sampled cells
    """
    batches = sp.unique(adata.obs['batch'])
    assert(len(batches)==2)
    adata_batch1 = adata[adata.obs['batch']==batches[0],:]
    adata_batch2 = adata[adata.obs['batch']==batches[1],:]
    # reduction of the cell number
    np.random.seed(random_state)
    n_cells_to_select = min(adata_batch1.shape[0], adata_batch2.shape[0], max_cells)
    if(n_cells_to_select!=adata_batch1.shape[0]):
        cells_to_select = np.random.randint(0, adata_batch1.shape[0], n_cells_to_select)
        adata_batch1 = adata_batch1[cells_to_select, :]
        adata_batch1.obs_names_make_unique()
    if(n_cells_to_select!=adata_batch2.shape[0]):
        cells_to_select = np.random.randint(0, adata_batch2.shape[0], n_cells_to_select)
        adata_batch2 = adata_batch2[cells_to_select, :]
        adata_batch2.obs_names_make_unique()
    adata_merged = anndata.AnnData.concatenate(adata_batch1, adata_batch2, join='inner', batch_key='batch')
    return(adata_merged)


def batch_correct(adata, method='reg'):
    """
    Function to perform batch correction on anndata object
    inputs:
        adata: anndata object with "sample" column in obs
        method: batch correction method {'reg','combat','mnn'}
    outputs: 
        dictionary with anndata objects with batch corrected values stored in .X
    """
    adata_batch = dict()
    adata = adata.copy()
    samples = sp.unique(adata.obs['sample'])
    for sample in samples:
        # skip samples that have no correspondence
        if(sample in ['sample35', 'sample67']):
            continue
        df = adata[adata.obs['sample']==sample].copy()
        df.X = np.array(pd.DataFrame(df.X).apply(lambda x: scale(x), axis=0))
        # batch correction
        if(method=='reg'):
            df_batch = sc.pp.regress_out(df, 'batch', copy=True)
        elif(method=='combat'):
            df_batch = df.copy()
            df_batch.X = sc.pp.combat(df_batch, key='batch', covariates=None, inplace=False)
        elif(method=="mnn"):
            batches = sp.unique(df.obs['batch'])
            assert(len(batches)==2)
            df_batch1 = df[df.obs['batch']==batches[0],:]
            df_batch2 = df[df.obs['batch']==batches[1],:]
            df_batch = sc.external.pp.mnn_correct(df_batch1, df_batch2, key='batch',
                                                       do_concatenate=True, n_jobs=1)[0]
        adata_batch[sample] = df_batch   
    return(adata_batch)

def prep_anndata_for_eval(adata, div_ent_dim=20, random_state=0):
    """
    Function to prepare anndata for eval using BERMUDA workflow
    inputs:
        adata: anndata
        div_ent_dim: #features to use for calculation of divergence score and entropy
                    (if smaller than adata.X.shape[1] then UMAP dim reduction with div_ent_dim codes performed)
        random_state: random seed
    outpus:
        umap_codes, data, cell_type_labels, batch_labels, number_of_datasets
    """
    df = adata.copy()
    batch_labels = df.obs['batch']
    cts_labels = df.obs['cell_type']
    df = df.X

    num_datasets = len(set(batch_labels))
    batch_dict = dict(zip(set(batch_labels), range(len(set(batch_labels)))))
    batch_labels_num = np.array([batch_dict[x]+1 for x in batch_labels])
    cts_dict = dict(zip(set(cts_labels), range(len(set(cts_labels)))))
    cts_labels_num = np.array([cts_dict[x]+1 for x in cts_labels])

    umap_code = cal_UMAP(df, div_ent_dim, random_state=random_state)
    return(umap_code, df, cts_labels_num, batch_labels_num, num_datasets)

def eval_batch_sample(adata_dict, random_state=345, umap_dim=50, div_ent_dim=50, sil_dim=50):
    np.random.seed(random_state)
    samples = adata_dict.keys()
    eval_batch = pd.DataFrame(columns=['sample','divergence_score', 'entropy_score', 'silhouette_score'],
                  index = list(samples))
    for sample in samples:
        adata_sample = adata_dict[sample].copy()
        try:
            umap_code, df, cts_labels_num, batch_labels_num, num_datasets = prep_anndata_for_eval(adata_sample)
            divergence_score, entropy_score, silhouette_score = evaluate_scores(umap_code, df, cts_labels_num,
                                                                                batch_labels_num, num_datasets,
                                                                                div_ent_dim, sil_dim, 'cosine', 
                                                                               random_state = random_state)  
        except:
            divergence_score, entropy_score, silhouette_score = np.nan, np.nan,np.nan
        eval_batch.loc[sample,:] = [sample,divergence_score, entropy_score, silhouette_score]
    return(eval_batch)


def load_sample_data(path, samples_selected=['sample1'], batch_names= ['batch1', 'batch2']):
    """
    Function to load and merge data for samples and convert it to a desired format
    """
    df_full = None
    for sample in samples_selected:
        x1_train, x1_test, x2_train, x2_test = load_data_basic(path,
                                     sample=sample, batch_names=batch_names, panel=None)

        tmp_  = pd.concat([x1_train, x2_train])
        if df_full is None:
            df_full = tmp_
        else:
            df_full = pd.concat([df_full, tmp_], axis = 0 )

    metadata_batch = [ i.split('_')[0] for i in df_full.index]
    metadata_cell = [ i.split('_')[-1] for i in df_full.index]
    metadata_sample = [ i.split('_')[1] for i in df_full.index]
    df_full['metadata_batch'] = metadata_batch
    df_full['metadata_celltype'] = metadata_cell
    df_full['metadata_sample'] = metadata_sample
    df_full = df_full.dropna(axis=1)
    df_full = df_full.reset_index(drop = True)
    return(df_full)

