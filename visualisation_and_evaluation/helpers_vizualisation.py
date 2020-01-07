import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import umap
reducer = umap.UMAP()
import matplotlib.cm as cm
import os
import scipy as sp


def plot_tsne(data, do_pca=True, n_plots=2, iter_=500, pca_components=11, save_as=None, folder_name='figures',
              random_state=345, modelname='', palette=None, legend=True):
    ''' 
    Function to generate t-sne plot 
    inputs: 
        data: cell x markers: has the labels as index!! eg. Data23_Panel1_tx_NR4_Patient9
        do_pca: performs pca prior to t-sne, no downsampling there
        n_plots: Tries different perplexity values, 
        iter_ : fitting 
        pca_components: PCs
    '''
    np.random.seed(random_state)
    Labels = list(data.index)
    if do_pca: 
        pca = PCA(n_components=pca_components, random_state=random_state)
        data = pca.fit_transform(data)
    for i in range(n_plots):
        perplexity_ = 10* (i + 1)
        tsne = TSNE(n_components=2,verbose=1,perplexity=perplexity_,n_iter=iter_, random_state=random_state)
        X_tsne = tsne.fit_transform(data)
        Xf = pd.DataFrame(X_tsne)
        Xf.columns = ["t-sne1", "t-sne2"]
        Xf['labels'] = Labels
        sns.lmplot("t-sne1", "t-sne2",hue="labels",data=Xf, fit_reg=False, scatter_kws={'alpha': 0.1}, palette=palette,
                   legend=legend)
        plt.title(modelname + ': t-SNEperplexity = {}, iter = {}'.format(perplexity_, iter_), fontsize=12)
        if save_as is not None:
            plt.savefig(os.path.join(folder_name, save_as+'_p'+str(perplexity_)), bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_metrics(metrics, fname, autoencoder=False):
    #metric_names = [m for m in metrics.keys() if m != 'epoch']
    plt.figure()
    plt.plot(metrics['epoch'], metrics['d_loss'], label='d_loss')
    plt.plot(metrics['epoch'], metrics['g_loss'], label='g_loss')
    plt.title('Generator vs Discriminator Cross-entropy')
    plt.legend()
    plt.grid()
    plt.savefig(fname + '_' + 'g_d_losses')
    plt.close()

    if autoencoder:
        plt.plot(metrics['epoch'], metrics['d_accuracy'], label='d_accuracy')
        plt.plot(metrics['epoch'], metrics['g_accuracy'], label='g_accuracy')
        plt.title('Generator vs Discriminator accuracy')
        plt.legend()
        plt.grid()
        plt.savefig(fname + '_' + 'g_d_accuracy')
        plt.close()

        plt.plot(metrics['epoch'], metrics['g_reconstruction_error'])
        plt.title('Generator mean absolute error')
        plt.grid()
        plt.savefig(fname + '_' + 'g_reconstruction_error')
        plt.close()

        plt.plot(metrics['epoch'], metrics['g_loss_total'])
        plt.title('Generator total loss (mae+xentropy, scaled)')
        plt.grid()
        plt.savefig(fname + '_' + 'g_loss_total')
        plt.close()


def plot_umap(data, random_state_=42, save_as=None, folder_name='figures', modelname=''):
    ''' 
    Function to generate Umap plot
    Inputs: 
        data: cell x markers: has the labels as index!! eg. Data23_Panel1_tx_NR4_Patient9
        random_state_ : parameters of umap 
    '''
    colors = ['r', 'b', 'g', 'c', 'm', 'o', 'y']
    reducer = umap.UMAP(random_state= random_state_)
    reducer.fit(data)
    embedding = reducer.transform(data)
    df_embedding = pd.DataFrame(embedding)
    df_embedding.columns = ["comp 1", "comp 2"]
    df_embedding['ID'] = data.index
    for i, row in enumerate(df_embedding.groupby("ID")):
        tbl = row[1]
        plt.scatter(tbl["comp 1"], tbl["comp 2"], c=colors[i], label = row[0], s=15, alpha=0.1)
        plt.xlabel('Umap 1', fontsize = 12)
        plt.ylabel('Umap 2', fontsize = 12)
        plt.title(modelname + ': UMAP projection', fontsize=12)
    plt.legend(np.unique(df_embedding["ID"]))
    if save_as is not None:
        plt.savefig(os.path.join(folder_name, save_as))
        plt.close()
    else:
        plt.show()
        

def plot_scores(data, xcol, ycol, title="Evaluation scores", save_as=None, folder_name='figures', ax=None, legend='brief'):
    """
    Function to plot all the scores from different batch correction methods
    data: pd dataframe with all the scores
    xcol: name of the column to plot on the xaxis
    ycol: name of the column to plot on the yaxis
    title: plot title
    """
    if(len(sp.unique(data['sample']))>1):
        score_plot = sns.scatterplot(x=xcol, y=ycol, 
                                     data=data,
                                     hue = 'method', style='sample', legend=legend, ax = ax)
    else:
        score_plot = sns.scatterplot(x=xcol, y=ycol, 
                                     data=data,
                                     hue = 'method', legend=legend, ax = ax)
    # correct labels
    xlab = score_plot.get_xlabel()
    xlab = xlab.replace('_',' ')
    ylab = score_plot.get_ylabel()
    ylab = ylab.replace('_',' ')
    score_plot.set(xlabel=xlab, ylabel=ylab, title = title)
    # move the legend outside the plot
    if(legend!=False):
        handles, names = score_plot.get_legend_handles_labels()
        score_plot.legend(handles, names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save_as is not None:
        plt.savefig(os.path.join(folder_name, save_as), bbox_inches='tight')
        plt.close()
    else:
        score_plot
        #plt.show()