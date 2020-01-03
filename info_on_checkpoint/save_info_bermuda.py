import os
import pandas as pd
from visualisation_and_evaluation.helpers_eval_bermuda import evaluate_scores, separate_metadata, prep_data_for_eval
import csv
import numpy as np


def save_dataframes(epoch, x1, x2, gx1, fname, dir_name='output'):
    if epoch == 0:
        x2 = pd.DataFrame(x2)
        x1 = pd.DataFrame(x1)
        # safe the original dataframes for batch 1 and batch 2 (x1 and x2)
        x1.to_csv(os.path.join(dir_name, fname, 'x1_epoch' + str(epoch) + '.csv'), index_label=False)
        x2.to_csv(os.path.join(dir_name, fname, 'x2_epoch' + str(epoch) + '.csv'), index_label=False)
    # save the dataframe for batch 1 transformed
    gx1 = pd.DataFrame(gx1)
    gx1.to_csv(os.path.join(dir_name, fname, 'gx1_epoch' + str(epoch) + '.csv'), index_label=False)


def save_scores(epoch, x1, x2, gx1, metrics, fname, x1_cells, x2_cells, cell_label, dir_name='output'):
    gx1 = pd.DataFrame(gx1)
    gx1.index = [str(i) + '_transformed' for i in gx1.index]
    x2 = pd.DataFrame(x2)
    #gx1.to_csv(os.path.join(dir_name, fname, 'gx1_epoch' + str(epoch) + '.csv'), index_label=False)

    df_eval = pd.concat([gx1, x2])
    df = df_eval

    batches = [1] * len(x1_cells) +  [2] * len(x2_cells)
    cells = list(np.concatenate([x1_cells, x2_cells]))
    prep = {'bermuda': cells, 'batch': batches}
    metadf = pd.DataFrame.from_dict(prep)
    metadf = metadf.merge(cell_label, on='bermuda', how='left')
    metadf = metadf.rename({'original': 'cell_type'}, axis = 1)

    #df, metadf = separate_metadata(df_eval)
    umap_codes, data, cell_type_labels, batch_labels, num_datasets = prep_data_for_eval(df, metadf, 50,
                                                                                        random_state=345)
    divergence_score, entropy_score, silhouette_score = evaluate_scores(umap_codes, data, cell_type_labels,
                                                                        batch_labels, num_datasets,
                                                                        50, 50, 'cosine',
                                                                        random_state=345)
    f = open(os.path.join(dir_name, fname, 'scores.csv'), 'a', newline='')
    score_writer = csv.writer(f)
    score_writer.writerow([epoch, divergence_score, entropy_score, silhouette_score])
    print('divergence:{}, entropy {}, silhouette {}'.format(divergence_score, entropy_score, silhouette_score))
    f.close()

    f = open(os.path.join(dir_name, fname, 'metrics.csv'), 'w', newline='')
    score_writer = csv.writer(f)
    score_writer.writerow(list(metrics.keys()))
    for metrics in np.transpose(list(metrics.values())):
        score_writer.writerow(metrics)
    f.close()
