import sys
import os
import numpy as np
sys.path.append('/Users/laurieprelot/Documents/Projects/2019_Deep_learning/git_DL')
datapath = '/Users/laurieprelot/Documents/Projects/2019_Deep_learning/data/Chevrier-et-al'


from loading_and_preprocessing.data_loader_bermuda import load_data_basic_bermuda

path = os.path.join(datapath, 'chevrier_data_pooled_full_panels.parquet')

### Load Module
x1, x2 = load_data_basic_bermuda(path, path_equivalence = '/Users/laurieprelot/Documents/Projects/2019_Deep_learning/git_DL/bermuda_gan/equivalence_tables',
                                 sample='sample5', batch_names=['batch1', 'batch3'], panel=None)