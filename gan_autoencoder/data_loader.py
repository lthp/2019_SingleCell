import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize(x):
    data = x.values
    data = 2 * (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)) - 1
    return pd.DataFrame(data=data, columns=x.columns, index=x.index)

def load_data_basic(path, patient='sample1', batch_names=['batch1', 'batch2'], seed=42,
                    n_cells_to_select=500):
    """
    Function to load data and split into 2 inputs with train and test sets
    inputs:
        path: path to the data file
        patient: name of the patient to consider
        batch_names: a list of batch names to split the data
        n_cells_to_select: number of cells to select for quicker runs, if 0 then all cells are selected
    outputs:
        x1_train, x1_test: train and test sets form the first batch
        x2_train, x2_test: train and test sets form the second batch
    """
    df = pd.read_parquet(path, engine='pyarrow')
    selected_cols = [col for col in df.columns if "metadata" not in col]
    df = df.loc[:, selected_cols]
    idx = df.index.get_values()
    x1_idx = [x for x in idx if patient in x and batch_names[0] in x and patient+'0' not in x][0]
    x1 = df.loc[x1_idx, :].copy()
    x2_idx = [x for x in idx if patient in x and batch_names[1] in x and patient+'0' not in x][0]
    x2 = df.loc[x2_idx, :].copy()
    if n_cells_to_select > 0:
        cells_to_select = np.random.uniform(0, x1.shape[0], n_cells_to_select)
        x1 = x1.iloc[cells_to_select, :]
        cells_to_select = np.random.uniform(0, x2.shape[0], n_cells_to_select)
        x2 = x2.iloc[cells_to_select, :]
    x1 = normalize(x1)
    x2 = normalize(x2)
    x1_train, x1_test = train_test_split(x1, test_size=0.2, random_state=42)
    x2_train, x2_test = train_test_split(x2, test_size=0.2, random_state=42)
    return x1_train, x1_test, x2_train, x2_test
