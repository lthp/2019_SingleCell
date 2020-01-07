import os
import pandas as pd
from visualisation_and_evaluation.helpers_vizualisation import plot_tsne, plot_umap
import seaborn as sns

c = sns.color_palette()
cell_dictionary = {'Myeloid': c[0],
                   'T': c[1],
                   'NK': c[2],
                   'Non': c[3],
                   'B': c[4],
                   'Plasma': c[5],
                   'Unknown': c[6],
                   'pDC': c[7],
                   'DC': c[8],
                   'Granulocytes': c[9]}

def generate_plots(epoch, x1, x2, gx1, fname, dir_name='figures', legend=True):
    #x1 = x1[:100]
    #x2 = x2[:100]
    #gx1 = gx1[:100]
    cells_x1 = [i.split('_')[3][8:].split('.')[0] for i in x1.index]
    cells_x2 = [i.split('_')[3][8:].split('.')[0] for i in x2.index]

    x1_plot = x1.copy()
    x2_plot = x2.copy()
    gx1_plot = gx1.copy()
    x1_plot.index = cells_x1  # Done to make plots look nicer
    x2_plot.index = cells_x2
    gx1_plot.index = cells_x1
    plot_tsne(pd.concat([x2_plot, gx1_plot]), do_pca=True, n_plots=1, iter_=500, pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_gx1_x2_celltypes'), folder_name=dir_name,
              modelname=fname + ' gx1-x2',
              palette=cell_dictionary,
              legend=legend
              )
    plot_tsne(pd.concat([x1_plot, gx1_plot]), do_pca=True, n_plots=1, iter_=500, pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_gx1_x1_celltypes'), folder_name=dir_name,
              modelname=fname + ' gx1-x1',
              palette=cell_dictionary,
              legend=legend
              )

    batch_x1 = x1.index[0].split('_')[0]
    batch_x2 = x2.index[0].split('_')[0]
    sample_x1 = x1.index[0].split('_')[1]
    sample_x2 = x2.index[0].split('_')[1]
    x1_plot.index = [batch_x1 + sample_x1 for i in range(len(x1_plot))]  # Done to make plots look nicer
    x2_plot.index = [batch_x2 + sample_x2 for i in range(len(x2_plot))]
    gx1_plot.index = [batch_x1 + sample_x1 + '_transformed' for i in range(len(gx1_plot))]
    plot_tsne(pd.concat([x2_plot, gx1_plot]), do_pca=True, n_plots=1, iter_=500,
              pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_gx1_x2_batches'), folder_name=dir_name,
              modelname=fname + ' gx1-x2',
              legend=legend
              )
    plot_tsne(pd.concat([x1_plot, gx1_plot]), do_pca=True, n_plots=1, iter_=500,
              pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_gx1_x1_batches'), folder_name=dir_name,
              modelname=fname + ' gx1-x1',
              legend=legend
              )


def generate_raw_plots(epoch, x1, x2, fname, dir_name='figures'):
    #x1 = x1[:100]
    #x2 = x2[:100]
    cells_x1 = [i.split('_')[3][8:].split('.')[0] for i in x1.index]
    cells_x2 = [i.split('_')[3][8:].split('.')[0] for i in x2.index]

    x1_plot = x1.copy()
    x2_plot = x2.copy()
    x1_plot.index = cells_x1  # Done to make plots look nicer
    x2_plot.index = cells_x2
    plot_tsne(pd.concat([x2_plot, x1_plot]), do_pca=True, n_plots=1, iter_=500, pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_x1_x2_celltypes'), folder_name=dir_name,
              modelname=fname + ' x1-x2',
              palette=cell_dictionary
              )

    batch_x1 = x1.index[0].split('_')[0]
    batch_x2 = x2.index[0].split('_')[0]
    sample_x1 = x1.index[0].split('_')[1]
    sample_x2 = x2.index[0].split('_')[1]
    x1_plot.index = [batch_x1 + sample_x1 for i in range(len(x1_plot))]  # Done to make plots look nicer
    x2_plot.index = [batch_x2 + sample_x2 for i in range(len(x2_plot))]
    plot_tsne(pd.concat([x2_plot, x1_plot]), do_pca=True, n_plots=1, iter_=500,
              pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_x1_x2_batches'), folder_name=dir_name,
              modelname=fname + ' x1-x2'
              )


def plot_chevrier():
    path = r'..\final_plots'
    df_dir = os.path.join(path, 'chevrier', 'dataframes_for_plotting')
    fig_dir = os.path.join(path, 'chevrier', 'final_plots')
    fnames = [f for f in os.listdir(df_dir) if 'gx1' in f and 'epoch0' not in f and 'bermuda' not in f]
    for fname in fnames:
        print(fname)
        epoch = fname.split('epoch')[1][:-4]
        modelname = fname.split('_gx1')[0]
        fgx1 = fname
        fx1 = fname.split('gx1')[0] + 'x1_epoch0.csv'
        fx2 = fname.split('gx1')[0] + 'x2_epoch0.csv'

        gx1 = pd.read_csv(os.path.join(df_dir, fgx1))
        x1_train_df = pd.read_csv(os.path.join(df_dir, fx1))
        x2_train_df = pd.read_csv(os.path.join(df_dir, fx2))
        # gx1 = gx1[:1000]
        # x1_train_df = x1_train_df[:1000]
        # x2_train_df = x2_train_df[:1000]
        print(gx1.shape)
        print(x1_train_df.shape)
        print(x2_train_df.shape)
        if not os.path.isdir(os.path.join(fig_dir, modelname)):
            os.makedirs(os.path.join(fig_dir, modelname))
        #generate_plots(epoch, x1_train_df, x2_train_df, gx1, fname=modelname,  dir_name=fig_dir)

    samples = ['5', '65', '75']
    for sample in samples:
        modelname='raw_sample_'+sample
        x1 = [i for i in os.listdir(df_dir) if '_' + sample + '_x1' in i][0]
        x2 = [i for i in os.listdir(df_dir) if '_' + sample + '_x2' in i][0]
        x1_train_df = pd.read_csv(os.path.join(df_dir, x1))
        x2_train_df = pd.read_csv(os.path.join(df_dir, x2))
        if not os.path.isdir(os.path.join(fig_dir, modelname)):
            os.makedirs(os.path.join(fig_dir, modelname))
        generate_raw_plots(0, x1_train_df, x2_train_df, fname=modelname, dir_name=fig_dir)


if __name__ == '__main__':
    plot_chevrier()
