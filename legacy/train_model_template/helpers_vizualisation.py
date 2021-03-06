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


def eval_knn_proportions(df, k=10):
    """
    Function to evaluate mixing of the batches
    inputs:
        df: a pandas dataframe with cells in rows and markers+metadata_markers in columns. 
        The IDs should be in the index. eg. Data23_Panel1_tx_NR4_Patient9
        (note: atm works with 2 batches only)
        k: number of nearest neighbors to consider
    outputs:
        batch1_proportions: a list containing proportions of first batch encountered in cell neighborhoods
    """
    col_w_metadata = [x for x in df.columns if 'metadata_' in x]
    X = np.array(df.loc[:,~df.columns.isin(col_w_metadata)])
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    batch1_name = list(set(df.index))[0]
    batch1_proportions = []
    for i in range(df.shape[0]):
        idx = indices[i]
        batch_counts = df.iloc[idx,:].reset_index()[df.iloc[idx,:].index.name].value_counts()
        batch_prop = batch_counts/len(idx)
        if(batch_prop.index[0]==batch1_name):
            batch_prop = batch_prop[0]
        else:
            batch_prop = 1 - batch_prop[0]
        batch1_proportions.append(batch_prop)
    return(batch1_proportions)


def plot_tsne(data, do_pca = True, n_plots = 2, iter_ = 500, pca_components = 20):
    ''' 
    Function to generate t-sne plot 
    inputs: 
        data: cell x markers: has the labels as index!! eg. Data23_Panel1_tx_NR4_Patient9
        do_pca: performs pca prior to t-sne, no downsampling there
        n_plots: Tries different perplexity values, 
        iter_ : fitting 
        pca_components: PCs'''
    Labels = list(data.index)
    if do_pca: 
        pca = PCA(n_components=pca_components)
        data = pca.fit_transform(data)
    for i in range(n_plots):
        perplexity_ = 10* (i + 1)
        tsne = TSNE(n_components=2,verbose=1,perplexity=perplexity_,n_iter=iter_)
        X_tsne = tsne.fit_transform(data)
        Xf = pd.DataFrame(X_tsne)
        Xf.columns = ["t-sne1","t-sne2"]
        Xf['labels'] = Labels
        sns.lmplot("t-sne1", "t-sne2",hue="labels",data=Xf, fit_reg=False)
        plt.title('Plot: t-SNE projection of the dataset perplexity = {}, iter = {}'.format(perplexity_, iter_), fontsize=15)
        plt.show()

def plot_umap(data, random_state_ = 42):
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
        plt.scatter(tbl["comp 1"], tbl["comp 2"], c=colors[i], label = row[0], s=15)
        plt.xlabel('Umap 1', fontsize = 12)
        plt.ylabel('Umap 2', fontsize = 12)
        plt.title('UMAP projection of the dataset with random state'.format(random_state_), fontsize=12)
    plt.legend(np.unique(df_embedding["ID"]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3, fancybox=True, shadow=True)
    plt.show()