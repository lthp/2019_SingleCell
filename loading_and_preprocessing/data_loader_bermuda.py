import numpy as np
import pandas as pd
import os


def make_normalization(x):
    data = x.values
    data = 2 * (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0)) - 1
    return pd.DataFrame(data=data, columns=x.columns, index=x.index)


def load_data_basic_bermuda(path, path_equivalence, sample='sample1', batch_names=['batch1', 'batch2'], panel=None , normalize = False):
    """
    Function to load data and split into 2 inputs with train and test sets
    inputs:
        path: path to the data file
        sample: name of the sample to consider
        batch_names: a list of batch names to split the data
        n_cells_to_select: number of cells to select for quicker runs, if 0 then all cells are selected (min of the 2 batches)
        test_size: proportion of the test set
    outputs:
        First and second batch normalized concatenated
        with rows 0-3
        dataset_label: 1 to 2
        metadata_sample: integer, equivalence with original name saved in equivalence table
        metadata_celltype   : integer, equivalence with original name saved in equivalence table
        metadata_phenograph : integer, equivalence with original name saved in equivalence table. The phenograph labels have been set as non overlapping between batches
    """


    df = pd.read_parquet(path, engine='pyarrow')
    if(panel is not None):
        df = df.loc[df['metadata_panel'].str.startswith(panel),:]
        # update batches names that are present in the panel
        panel_batch_names = list(df.loc[:,'metadata_batch'].unique())
        if(len([x for x in batch_names if x not in panel_batch_names])):
            batch_names = panel_batch_names


    # Remove columns with ann (bcs of merging the panels)
    df = df.dropna(axis=1)
    df = df.reset_index(drop = True)



    # Replace the metadata with integers
    metadata = [ 'metadata_sample', 'metadata_celltype', 'metadata_phenograph' ]
    metadata_extended = ['dataset_label'] + metadata
    equivalence_table = {}
    for field in metadata:
        equivalence_table[field] = {}
        for i, j in enumerate(np.unique(df.loc[:,field])):
            df.loc[df[field] == j , field] = i
            equivalence_table[field][j] = i
        eq = pd.DataFrame.from_dict(equivalence_table[field], orient='index').reset_index()
        eq.columns =  ['original', 'bermuda']
        if field not in 'metadata_phenograph':
            eq.to_csv(os.path.join(path_equivalence, 'equivalence_table_' + field + '.tsv'), sep = '\t', index = None, header = True )
        else:
            cluster_tbl = eq
        if field == 'metadata_sample':
            sample = int(eq.loc[eq['original'] == sample, 'bermuda'])



    # Extract batches
    if('metadata_batch' in df.columns and 'metadata_sample' in df.columns):
        x1 = df.loc[(df['metadata_batch']==batch_names[0]) & (df['metadata_sample']==sample),:].copy()
        x2 = df.loc[(df['metadata_batch']==batch_names[1]) & (df['metadata_sample']==sample),:].copy()
    else:
        idx = df.index.get_values()
        x1_idx = [x for x in idx if sample in x and batch_names[0] in x and sample+'0' not in x]
        x1_idx = [i for (i,t) in enumerate(idx) if t in x1_idx]
        x1 = df.loc[x1_idx, :].copy()
        x2_idx = [x for x in idx if sample in x and batch_names[1] in x and sample+'0' not in x]
        x2_idx = [i for (i,t) in enumerate(idx) if t in x2_idx]
        x2 = df.loc[x2_idx, :].copy()

    selected_cols = [col for col in df.columns if "metadata" not in col]
    batch_dict = {batch_names[0]: x1, batch_names[1]: x2}

    cluster_idx = 1
    dataset_id = 1
    joined_batch = None
    for batch_name, batch_values in batch_dict.items():
        # Shift cluster names
        cluster_labels = cluster_tbl.copy()
        cluster_labels['bermuda'] = 'NaN'
        clusters_uq = np.unique(batch_values['metadata_phenograph'])
        if dataset_id == 2:
            for i, clu in enumerate(clusters_uq):
                cluster_labels.loc[cluster_labels['original'] == float(clu), 'bermuda'] = i + cluster_idx
                batch_values.loc[batch_values['metadata_phenograph'] == float(clu)  , 'metadata_phenograph'] = i + cluster_idx
        cluster_idx += len(clusters_uq)
        cluster_labels.to_csv(os.path.join(path_equivalence, 'equivalence_table_' + 'metadata_phenograph_'+ batch_name + '.tsv'), sep='\t', index=None,
                  header=True)
        # remove metadata columns
        x_mx = batch_values.loc[:, selected_cols]
        if normalize:
            x_mx = make_normalization(x_mx)
        batch_values['dataset_label'] = dataset_id
        batch_values_save = pd.concat([batch_values.loc[:, metadata], x_mx], axis=1)

        # save individual batch
        print(batch_name)
        print(batch_values_save.shape[0] - 3 )
        print(np.unique(batch_values['metadata_phenograph']))
        batch_values_save = batch_values_save.transpose()
        batch_values_save.to_csv(os.path.join(os.path.dirname(path),
                                         'chevrier_data_pooled_full_panels.' +
                                             batch_name + '.bermuda' + '.tsv'),
                             index = True, sep = '\t', header = False)
        # Create joined table
        batch_values_save = pd.concat([batch_values.loc[:, metadata_extended], x_mx], axis=1)
        dataset_id+=1


        if joined_batch is not None:
            joined_batch = pd.concat([joined_batch, batch_values_save], axis = 0, sort = False)
            joined_batch = joined_batch.reset_index(drop = True)
            joined_batch = joined_batch.transpose()
            joined_batch.to_csv(os.path.join(os.path.dirname(path),
                                             'chevrier_data_pooled_full_panels.' + '_'.join(batch_names) + '.bermuda' + '.tsv'),
                                    index=True, sep='\t', header=False) #foo  = pd.read_parquet(pq, engine='pyarrow')
            print('joined batch is of size: ')
            print(joined_batch.shape)
            print(joined_batch.head())
        else:
            joined_batch = batch_values_save



