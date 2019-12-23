import os
import pandas as pd
from visualisation_and_evaluation.helpers_vizualisation import plot_tsne, plot_metrics, plot_umap


def plot_progress(epoch, x1, x2, gx1, metrics, fname, dir_name='figures', tsne=True, umap=True, autoencoder=True,
                  modelname='aegan'):
    plot_metrics(metrics, os.path.join(dir_name, fname, 'metrics'), autoencoder=autoencoder)
    batch_x1 = x1.index[0].split('_')[0]
    batch_x2 = x2.index[0].split('_')[0]
    sample_x1 = x1.index[0].split('_')[1]
    sample_x2 = x2.index[0].split('_')[1]
    x1_plot = x1.copy()
    x2_plot = x2.copy()
    x1_plot.index = [batch_x1 + sample_x1 for i in range(len(x1_plot))]  # Done to make plots look nicer
    x2_plot.index = [batch_x2 + sample_x2 for i in range(len(x2_plot))]
    if epoch == 0:
        if tsne:
            plot_tsne(pd.concat([x1_plot, x2_plot]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                      save_as=os.path.join(fname, modelname + '_tsne_x1-x2_epoch' + str(epoch)), folder_name=dir_name)
        if umap:
            plot_umap(pd.concat([x1_plot, x2_plot]), save_as=os.path.join(fname, modelname + '_umap_x1-x2_epoch' + str(epoch)),
                      folder_name=dir_name)

    gx1 = pd.DataFrame(data=gx1, columns=x1_plot.columns, index=x1_plot.index + '_transformed')
    if tsne:
        plot_tsne(pd.concat([x1_plot, gx1]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                  save_as=os.path.join(fname, modelname + '_tsne_x1-gx1_epoch' + str(epoch)), folder_name=dir_name)
        plot_tsne(pd.concat([gx1, x2_plot]), do_pca=True, n_plots=2, iter_=500, pca_components=20,
                  save_as=os.path.join(fname, modelname + '_tsne_gx1-x2_epoch' + str(epoch)), folder_name=dir_name)
    if umap:
        plot_umap(pd.concat([x1_plot, gx1]), save_as=os.path.join(fname, modelname + '_umap_gx1-x1_epoch' + str(epoch)),
                  folder_name=dir_name)
        plot_umap(pd.concat([x2_plot, gx1]), save_as=os.path.join(fname, modelname + '_umap_gx1-x2_epoch' + str(epoch)),
                  folder_name=dir_name)
