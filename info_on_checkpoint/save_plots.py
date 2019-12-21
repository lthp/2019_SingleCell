import os
import pandas as pd
from visualisation_and_evaluation.helpers_vizualisation import plot_tsne, plot_metrics, plot_umap


def plot_progress(epoch, x1, x2, gx1, metrics, fname, dir_name='figures', tsne=True, umap=True):
    plot_metrics(metrics, os.path.join(dir_name, fname, 'metrics'), autoencoder=True)
    x1.index = ['batch1_sample1' for i in range(len(x1))]  # Done to make plots look nicer
    x2.index = ['batch2_sample1' for i in range(len(x2))]
    if epoch == 0:
        if tsne:
            plot_tsne(pd.concat([x1, x2]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                      save_as=os.path.join(fname, 'aegan_tsne_x1-x2_epoch' + str(epoch)), folder_name=dir_name)
        if umap:
            plot_umap(pd.concat([x1, x2]), save_as=os.path.join(fname, 'aegan_umap_x1-x2_epoch' + str(epoch)),
                      folder_name=dir_name)

    gx1 = pd.DataFrame(data=gx1, columns=x1.columns, index=x1.index + '_transformed')
    if tsne:
        plot_tsne(pd.concat([x1, gx1]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                  save_as=os.path.join(fname, 'aegan_tsne_x1-gx1_epoch' + str(epoch)), folder_name=dir_name)
        plot_tsne(pd.concat([gx1, x2]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                  save_as=os.path.join(fname, 'aegan_tsne_gx1-x2_epoch' + str(epoch)), folder_name=dir_name)
    if umap:
        plot_umap(pd.concat([x1, gx1]), save_as=os.path.join(fname, 'aegan_umap_gx1-x1_epoch' + str(epoch)),
                  folder_name=dir_name)
        plot_umap(pd.concat([x2, gx1]), save_as=os.path.join(fname, 'aegan_umap_gx1-x2_epoch' + str(epoch)),
                  folder_name=dir_name)
