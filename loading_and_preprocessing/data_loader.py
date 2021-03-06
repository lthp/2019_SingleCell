import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def normalize(x):
    data = x.values
    data = 2 * (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0)) - 1
    return pd.DataFrame(data=data, columns=x.columns, index=x.index)


def load_data_basic(path, sample='sample1', batch_names=['batch1', 'batch2'],
                    panel=None, seed=42, n_cells_to_select=0, test_size=0.2, upsample=True):
    """
    Function to load data and split into 2 inputs with train and test sets
    inputs:
        path: path to the data file
        sample: name of the sample to consider
        batch_names: a list of batch names to split the data
        n_cells_to_select: number of cells to select for quicker runs, if 0 then all cells are selected (min of the 2 batches)
        test_size: proportion of the test set
    outputs:
        x1_train, x1_test: train and test sets form the first batch
        x2_train, x2_test: train and test sets form the second batch
    """
    df = pd.read_parquet(path, engine='pyarrow')
    if(panel is not None):
        df = df.loc[df['metadata_panel'].str.startswith(panel),:]
        # update batches names that are present in the panel
        panel_batch_names = list(df.loc[:,'metadata_batch'].unique())
        if(len([x for x in batch_names if x not in panel_batch_names])):
            batch_names = panel_batch_names
    # remove columns with ann (bcs of merging the panels)
    df = df.dropna(axis=1)
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
    # remove metadata columns
    selected_cols = [col for col in df.columns if "metadata" not in col]
    x1 = x1.loc[:, selected_cols]
    x2 = x2.loc[:, selected_cols]
    if n_cells_to_select > 0:  # Downsample
        n_cells_to_select = np.min([n_cells_to_select, x1.shape[0], x2.shape[0]])
        x1 = x1.sample(n=n_cells_to_select, replace=False)
        x2 = x2.sample(n=n_cells_to_select, replace=False)
    if upsample:
        if x1.shape[0] < x2.shape[0]:
            x1 = x1.sample(n=x2.shape[0], replace=True)
        elif x2.shape[0] < x1.shape[0]:
            x2 = x2.sample(n=x1.shape[0], replace=True)
    x1 = normalize(x1)
    x2 = normalize(x2)
    x1_train, x1_test = train_test_split(x1, test_size=test_size, random_state=seed)
    x2_train, x2_test = train_test_split(x2, test_size=test_size, random_state=seed)
    print('x1 shape', x1_train.shape)
    print('x2 shape', x1_train.shape)
    return x1_train, x1_test, x2_train, x2_test


def load_data_cytof(path, patient_id='rcc7', n=None, upsample=True):
    full = pd.read_parquet(path, engine='pyarrow')
    select_cols = [col for col in full.columns if "metadata" not in col]  # not include metadata
    select_cols.append('metadata_panel')
    full = full.loc[:, select_cols]
    panels = full.metadata_panel.unique()
    full_panel1 = full.loc[full['metadata_panel'] == panels[0]]
    # full_panel2 = full.loc[full['metadata_panel'] == panels[1]]

    # start working with batches and patients in panel1 only
    full_panel1 = full_panel1.dropna(how='all', axis='columns')
    full_panel1 = full_panel1.loc[:, ]
    full_patient = full_panel1.reset_index()
    full_patient = full_patient.rename({'level_0': 'batch', 'level_1': 'patient', 'level_2': 'cell'}, axis=1)
    full_patient = full_patient.loc[full_patient['patient'] == patient_id, :]
    batches = full_patient.batch.unique()

    full_patient_batch1 = full_patient.loc[full_patient['batch'] == batches[0]]  # split into the 2 batches
    full_patient_batch2 = full_patient.loc[full_patient['batch'] == batches[1]]  # for this patient

    if upsample:
        if len(full_patient_batch1) < len(full_patient_batch2):
            full_patient_batch1 = full_patient_batch1.sample(n=len(full_patient_batch2), replace=True)
        elif len(full_patient_batch2) < len(full_patient_batch1):
            full_patient_batch2 = full_patient_batch2.sample(n=len(full_patient_batch1), replace=True)

    if n is not None:
        full_patient_batch1 = shuffle(full_patient_batch1)
        full_patient_batch2 = shuffle(full_patient_batch2)
        full_patient_batch1 = full_patient_batch1.iloc[:n, :]
        full_patient_batch2 = full_patient_batch2.iloc[:n, :]

    # y = full_patient_batch1["batch"]  # the label is batch1 (the reference batch)
    x1 = full_patient_batch1.drop(["batch", 'cell', 'patient', 'metadata_panel'], axis=1)  # remove all but the markers as the data
    x2 = full_patient_batch2.drop(["batch", 'cell', 'patient', 'metadata_panel'], axis=1)
    x1 = normalize(x1)
    x2 = normalize(x2)
    x1.index = ['batch1_sample1' for i in range(len(x1))]   # Done to make plots look nicer
    x2.index = ['batch2_sample1' for i in range(len(x2))]

    x1_train, x1_test = train_test_split(x1, test_size=0.2, random_state=42)
    x2_train, x2_test = train_test_split(x2, test_size=0.2, random_state=42)
    return x1_train, x1_test, x2_train, x2_test
