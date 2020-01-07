import os
import pandas as pd
from visualisation_and_evaluation.helpers_eval import evaluate_scores, separate_metadata, prep_data_for_eval
import csv
import numpy as np


def save_dataframes(epoch, x1, x2, gx1, fname, dir_name='output', model_description=''):
    if epoch == 0:
        # safe the original dataframes for batch 1 and batch 2 (x1 and x2)
        x1.to_csv(os.path.join(dir_name, fname, model_description + '_x1_epoch' + str(epoch) + '.csv'),
                  index_label=False)
        x2.to_csv(os.path.join(dir_name, fname, model_description + '_x2_epoch' + str(epoch) + '.csv'),
                  index_label=False)
    # save the dataframe for batch 1 transformed
    gx1 = pd.DataFrame(data=gx1, columns=x1.columns, index=x1.index + '_transformed')
    gx1.to_csv(os.path.join(dir_name, fname, model_description + '_gx1_epoch' + str(epoch) + '.csv'), index_label=False)


def save_scores(epoch, x1, x2, gx1, metrics, fname, dir_name='output', model_description=''):
    gx1 = pd.DataFrame(data=gx1, columns=x1.columns, index=x1.index + '_transformed')
    # gx1.to_csv(os.path.join(dir_name, fname, 'gx1_epoch' + str(epoch) + '.csv'), index_label=False)
    df_eval = pd.concat([gx1, x2])
    df, metadf = separate_metadata(df_eval)
    umap_codes, data, cell_type_labels, batch_labels, num_datasets = prep_data_for_eval(df, metadf, 50,
                                                                                        random_state=345)
    divergence_score, entropy_score, silhouette_score = evaluate_scores(umap_codes, data, cell_type_labels,
                                                                        batch_labels, num_datasets,
                                                                        50, 50, 'cosine',
                                                                        random_state=345)
    f = open(os.path.join(dir_name, fname, 'scores_' + model_description+'.csv'), 'a', newline='')
    score_writer = csv.writer(f)
    score_writer.writerow([epoch, divergence_score, entropy_score, silhouette_score])
    print(divergence_score, entropy_score, silhouette_score)
    f.close()

    f = open(os.path.join(dir_name, fname, 'metrics_' + model_description+'.csv'), 'w', newline='')
    score_writer = csv.writer(f)
    score_writer.writerow(list(metrics.keys()))
    for metrics in np.transpose(list(metrics.values())):
        score_writer.writerow(metrics)
    f.close()
