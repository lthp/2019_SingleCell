import warnings
import pdb
import os
import numpy as np
import scipy as sp
import pandas as pd
from functools import reduce
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
import random

from absl import flags, app
flags.DEFINE_integer('n_samples', 10, 'number of patients')
flags.DEFINE_integer('n_markers', 20, 'number of markers')
flags.DEFINE_integer('n_cells_min', 1000,'min number of cells per patients (number drawn from [min,max])')
flags.DEFINE_integer('n_cells_max', 2000,'max number of cells per patients (number drawn from [min,max])')
flags.DEFINE_integer('n_batches',2,'number of batches')
flags.DEFINE_string('distribution', 'gamma', 'distribution to sample from {gamma, poisson}')
flags.DEFINE_integer('seed', 234, 'set random seed')
flags.DEFINE_integer('subset', 0, 'how many cell-types remove form each batch to create a set with different cell populations')
flags.DEFINE_string('path_save', None,'path to save the generated data')
flags.DEFINE_bool('add_ri_patient', False, 'whether to add a RI per patient to achieve hierarchical structure')

FLAGS = flags.FLAGS

def simulate_data(n_samples, n_markers, n_cells_min, n_cells_max,
                  n_batches, distribution, seed=234, add_ri_patient=False):
    """
    Function to simulate toy dataset with differences between batches
    inputs:
        n_samples: number of samples
        n_markers: number of markers
        n_cells_min, n_cells_max: min and max number of cells per patient (number drawn from [min, max])
        n_batches: number of batches (recommended 2)
        distribution: "gamma" and "poisson" allowed
        add_ri_patient: whether to add a Random Intercept per patient
    outputs:
        toy dataset with sample_id in "metadata_sample" columns and batch_id in "metadata_batch" columns
    """
    np.random.seed(seed)
    marker_names = ['X'+str(x+1) for x in range(n_markers)]
    sample_names = ['patient'+str(x+1) for x in range(n_samples)]    
    trans_shift = np.random.uniform(0,1,n_markers)
    trans_grad = np.random.uniform(1,2,n_markers)
    
    data_dict = dict()
    for b in range(n_batches):
        batch_name = 'batch'+str(b+1)
        np.random.seed(seed*(b+1))
        data_dict[batch_name] = dict()

        for s in range(n_samples):
            sample_name = 'sample'+str(s+1)
            n_cells = int(np.random.uniform(n_cells_min, n_cells_max, 1)[0])
            if(distribution=='gamma'):
                df = pd.DataFrame(np.random.gamma(2+b, 2+b, size=n_markers*n_cells).reshape(n_cells, n_markers))
            else:
                df = pd.DataFrame(np.random.poisson(1+b, size=n_markers*n_cells).reshape(n_cells, n_markers))
            df = df * trans_grad + trans_shift
            if(add_ri_patient):
                ri = np.random.normal(0,1,1)[0]
                df = df+ri
            df.columns = marker_names
            df['metadata_sample'] = sample_name
            df['metadata_batch'] = batch_name
            data_dict[batch_name][sample_name] = df
        data_dict[batch_name] = pd.concat(data_dict[batch_name], axis=0)

    toy_data = pd.concat(data_dict, axis=0)
    idx = [x+'_'+y for x,y in zip([str(x) for x in toy_data.index.get_level_values(0)],
                                [str(x) for x in toy_data.index.get_level_values(1)])]
    toy_data.index = idx
    #toy_data.drop(columns=['metadata_sample', 'metadata_batch'])
    toy_data.index = range(toy_data.shape[0])
    return(toy_data)

def main(argv):
    del argv
    
    toy_data = simulate_data(FLAGS.n_samples, FLAGS.n_markers, FLAGS.n_cells_min,
                             FLAGS.n_cells_max, FLAGS.n_batches, 
                             FLAGS.distribution, FLAGS.seed, FLAGS.add_ri_patient)
    # create index and change "sample" into cell-type
    toy_data['metadata_celltype'] = [x.replace('sample','type') for x in toy_data['metadata_sample']]
    toy_data['metadata_sample'] = 'sample1'
    idx = ['_'.join([a,b,c]) for a,b,c in zip(toy_data['metadata_batch'],
                                                  toy_data['metadata_sample'],
                                                  ['celltype'+x for x in toy_data['metadata_celltype']])]
    toy_data.index = idx
    if(FLAGS.subset>0):
        toy_data_sub = toy_data.copy()
        toy_data_sub['metadata_joint'] = [x+'_'+y for x,y in zip(toy_data_sub['metadata_batch'],
                                                                 toy_data_sub['metadata_celltype'])]
        to_remove = ['batch1_type2', 'batch1_type5', 'batch2_type3', 'batch2_type7']
        toy_data = toy_data.loc[~toy_data_sub['metadata_joint'].isin(to_remove),:]
    table = pa.Table.from_pandas(toy_data)
    pq.write_table(table, FLAGS.path_save)
    
if __name__ == '__main__':
    app.run(main)
