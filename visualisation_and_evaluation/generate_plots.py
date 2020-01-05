import os
import pandas as pd
from visualisation_and_evaluation.helpers_vizualisation import plot_tsne, plot_umap

def generate_plots(epoch, x1, x2, gx1, fname, dir_name='figures'):
    if not os.path.isdir(os.path.join(dir_name, fname)):
        os.makedirs(os.path.join(dir_name, fname))
    if 'bermuda' in fname:
        batch_x1 = 'batch1'
        batch_x2 = 'batch3'
        sample_x1 = 'sample5'
        sample_x2 = 'sample5'
    else:
        batch_x1 = x1.index[0].split('_')[0]
        batch_x2 = x2.index[0].split('_')[0]
        sample_x1 = x1.index[0].split('_')[1]
        sample_x2 = x2.index[0].split('_')[1]
    x1_plot = x1.copy()
    x2_plot = x2.copy()
    gx1_plot = gx1.copy()
    x1_plot.index = [batch_x1 + sample_x1 for i in range(len(x1_plot))]  # Done to make plots look nicer
    x2_plot.index = [batch_x2 + sample_x2 for i in range(len(x2_plot))]
    gx1_plot.index = [batch_x1 + sample_x1 + '_transformed' for i in range(len(gx1_plot))]

    if 'bermuda' in fname:
        plot_tsne(pd.concat([x1_plot, x2_plot]), do_pca=True, n_plots=2, iter_=500,
                  pca_components=min(x1.shape[1], 20),
                  save_as=os.path.join(fname, fname + '_tsne_x1-gx1_epoch' + str(epoch)), folder_name=dir_name,
                  modelname=fname + ' x1-x2')
        plot_umap(pd.concat([x1_plot, x2_plot]),
                  save_as=os.path.join(fname, fname + '_umap_gx1-x1_epoch' + str(epoch)),
                  folder_name=dir_name, modelname=fname + 'x1 - x2')

    plot_tsne(pd.concat([x1_plot, gx1_plot]), do_pca=True, n_plots=2, iter_=500, pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_tsne_x1-gx1_epoch' + str(epoch)), folder_name=dir_name,
              modelname=fname + ' x1 - gx1')
    plot_tsne(pd.concat([x2_plot, gx1_plot]), do_pca=True, n_plots=2, iter_=500, pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_tsne_gx1-x2_epoch' + str(epoch)), folder_name=dir_name,
              modelname=fname + 'x2 - gx1')

    plot_umap(pd.concat([x1_plot, gx1_plot]), save_as=os.path.join(fname, fname + '_umap_gx1-x1_epoch' + str(epoch)),
              folder_name=dir_name, modelname=fname + ' x1 - gx1')
    plot_umap(pd.concat([x2_plot, gx1_plot]), save_as=os.path.join(fname, fname + '_umap_gx1-x2_epoch' + str(epoch)),
              folder_name=dir_name, modelname=fname + ' x2 - gx1')

def generate_cell_type_plots(epoch, x1, x2, gx1, fname, dir_name='figures'):
    if not os.path.isdir(os.path.join(dir_name, fname)):
        os.makedirs(os.path.join(dir_name, fname))


    cells_x1 = [i.split('_')[3][8:].split('.')[0] for i in x1.index]
    cells_x2 = [i.split('_')[3][8:].split('.')[0] for i in x2.index]

    x1_plot = x1.copy()
    x2_plot = x2.copy()
    gx1_plot = gx1.copy()
    x1_plot.index = cells_x1  # Done to make plots look nicer
    x2_plot.index = cells_x2
    gx1_plot.index = cells_x1
    #x1_plot = x1_plot[:100]
    #gx1_plot = gx1_plot[:100]
    #x2_plot = x2_plot[:100]
    plot_tsne(pd.concat([x1_plot, gx1_plot]), do_pca=True, n_plots=1, iter_=500, pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_tsne_x1-gx1_epoch' + str(epoch)),
              folder_name=dir_name,
              modelname=fname + ' x1 - gx1')
    plot_tsne(pd.concat([x2_plot, gx1_plot]), do_pca=True, n_plots=1, iter_=500, pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_tsne_gx1-x2_epoch' + str(epoch)), folder_name=dir_name,
              modelname=fname + 'x2 - gx1')
    plot_tsne(gx1_plot, do_pca=True, n_plots=1, iter_=500, pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_tsne_gx1' + str(epoch)), folder_name=dir_name,
              modelname=fname + 'x2 - gx1')
    #plot_umap(pd.concat([x1_plot, gx1_plot]), save_as=os.path.join(fname, fname + '_umap_gx1-x1_epoch' + str(epoch)),
    #          folder_name=dir_name, modelname=fname + ' x1 - gx1')
    #plot_umap(pd.concat([x2_plot, gx1_plot]), save_as=os.path.join(fname, fname + '_umap_gx1-x2_epoch' + str(epoch)),
    #          folder_name=dir_name, modelname=fname + ' x2 - gx1')

def generate_raw_plots(epoch, x1, x2, fname, dir_name='figures'):
    if not os.path.isdir(os.path.join(dir_name, fname)):
        os.makedirs(os.path.join(dir_name, fname))
    if 'bermuda' in fname:
        batch_x1 = 'batch1'
        batch_x2 = 'batch3'
        sample_x1 = 'sample5'
        sample_x2 = 'sample5'
    else:
        batch_x1 = x1.index[0].split('_')[0]
        batch_x2 = x2.index[0].split('_')[0]
        sample_x1 = x1.index[0].split('_')[1]
        sample_x2 = x2.index[0].split('_')[1]
    x1_plot = x1.copy()
    x2_plot = x2.copy()
    x1_plot.index = [batch_x1 + sample_x1 for i in range(len(x1_plot))]  # Done to make plots look nicer
    x2_plot.index = [batch_x2 + sample_x2 for i in range(len(x2_plot))]

    plot_tsne(pd.concat([x1_plot, x2_plot]), do_pca=True, n_plots=2, iter_=500,
              pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_tsne_x1-gx1_epoch' + str(epoch)), folder_name=dir_name,
              modelname=fname + ' x1-x2')
    plot_umap(pd.concat([x1_plot, x2_plot]),
              save_as=os.path.join(fname, fname + '_umap_gx1-x1_epoch' + str(epoch)),
              folder_name=dir_name, modelname=fname + ' x1-x2')

def generate_raw_plots_celltypes(epoch, x1, x2, fname, dir_name='figures'):
    if not os.path.isdir(os.path.join(dir_name, fname)):
        os.makedirs(os.path.join(dir_name, fname))

    cells_x1 = [i.split('_')[3][8:].split('.')[0] for i in x1.index]
    cells_x2 = [i.split('_')[3][8:].split('.')[0] for i in x2.index]

    x1_plot = x1.copy()
    x2_plot = x2.copy()
    x1_plot.index = cells_x1
    x2_plot.index = cells_x2

    plot_tsne(pd.concat([x1_plot, x2_plot]), do_pca=True, n_plots=1, iter_=500,
              pca_components=min(x1.shape[1], 20),
              save_as=os.path.join(fname, fname + '_tsne_x1-x2_epoch' + str(epoch)), folder_name=dir_name,
              modelname=fname + ' x1-x2')

def plot_batches():
    path = r'C:\Users\heida\Documents\ETH\Deep Learning\final_plots'
    dtypes = ['chevrier', os.path.join('simulated', 'main')]

    for dtype in dtypes:
        df_dir = os.path.join(path, dtype, 'dataframes_for_plotting')
        fig_dir = os.path.join(path, dtype, 'final_plots2')

        fnames = [f for f in os.listdir(df_dir) if 'gx1' in f and 'epoch0' not in f]
        print(fnames)
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
            #gx1 = gx1[:1000]
            #x1_train_df = x1_train_df[:1000]
            #x2_train_df = x2_train_df[:1000]
            if not os.path.isdir(os.path.join(fig_dir, modelname)):
                os.makedirs(os.path.join(fig_dir, modelname))
            generate_plots(epoch, x1_train_df, x2_train_df, gx1, fname=modelname,  dir_name=fig_dir)


def plot_raw():
    path = r'C:\Users\heida\Documents\ETH\Deep Learning\final_plots'
    dtypes = ['chevrier', os.path.join('simulated', 'main')]

    dtype = 'chevrier'
    df_dir = os.path.join(path, dtype, 'dataframes_for_plotting')
    fig_dir = os.path.join(path, dtype, 'final_plots2')
    fig_dir_celltypes = os.path.join(path, dtype, 'final_plots_cell_types')
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
        generate_raw_plots_celltypes(0, x1_train_df, x2_train_df, fname=modelname, dir_name=fig_dir_celltypes)

    dtype = os.path.join('simulated', 'main')
    df_dir = os.path.join(path, dtype, 'dataframes_for_plotting')
    fig_dir = os.path.join(path, dtype, 'final_plots2')
    sample = '1'
    modelname='raw_sample_'+sample
    x1 = [i for i in os.listdir(df_dir) if '_x1_' in i][0]
    x2 = [i for i in os.listdir(df_dir) if '_x2_' in i][0]
    x1_train_df = pd.read_csv(os.path.join(df_dir, x1))
    x2_train_df = pd.read_csv(os.path.join(df_dir, x2))
    if not os.path.isdir(os.path.join(fig_dir, modelname)):
        os.makedirs(os.path.join(fig_dir, modelname))
    generate_raw_plots(0, x1_train_df, x2_train_df, fname=modelname, dir_name=fig_dir)


def plot_cell_types():
    path = r'C:\Users\heida\Documents\ETH\Deep Learning\final_plots'
    dtypes = ['chevrier', os.path.join('simulated', 'main')]

    for dtype in dtypes:
        df_dir = os.path.join(path, dtype, 'dataframes_for_plotting')
        fig_dir = os.path.join(path, dtype, 'final_plots_cell_types')

        fnames = [f for f in os.listdir(df_dir) if 'gx1' in f and 'epoch0' not in f and 'bermuda' not in f]
        print(fnames)
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
            #gx1 = gx1[:1000]
            #x1_train_df = x1_train_df[:1000]
            #x2_train_df = x2_train_df[:1000]
            print(gx1.shape)
            print(x1_train_df.shape)
            print(x2_train_df.shape)
            if not os.path.isdir(os.path.join(fig_dir, modelname)):
                os.makedirs(os.path.join(fig_dir, modelname))
            generate_cell_type_plots(epoch, x1_train_df, x2_train_df, gx1, fname=modelname,  dir_name=fig_dir)

if __name__ == '__main__':
    #plot_batches()
    plot_raw()
    #plot_cell_types()
