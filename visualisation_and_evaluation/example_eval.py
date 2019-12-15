import os
from visualisation_and_evaluation.helpers_eval import cal_UMAP, entropy, cal_entropy, evaluate_scores, separate_metadata, prep_data_for_eval

### some data downsampled to show the utility
df = pd.read_parquet('chevrier_data_pooled_nona.parquet')

batch_labels = df['metadata_batch'].tolist()
batch_unique = list(set(batch_labels))
batch_dict = dict(zip(batch_unique,['batch'+str(i+1) for i in range(len(batch_unique))]))
batch_labels = [batch_dict[x] for x in batch_labels]
cts_labels = ['celltype'+str(int(x)) for x in df['metadata_phenograph']]
df.index = ['_'.join([a,b,c]) for a, b, c in zip(batch_labels, df['metadata_sample'], cts_labels)]
random_state = 345
np.random.seed(random_state)
selected_cells = np.random.choice(range(df.shape[0]), 1000)
df = df.iloc[selected_cells,:]
df = df.loc[:,~df.columns.str.startswith('meta')]

##############################   exemplary workflow  ########################################
random_state = 345
np.random.seed(random_state)
umap_pca_dim = 50
div_ent_dim = 50
sil_dim = 50 #df.shape[1]

### df is the model output (a pandas dataframe with 1 index level, all metainfo separated by '_')
df, metadf = separate_metadata(df)
### prepare data for evaluation
umap_codes, data, cell_type_labels, batch_labels, num_datasets = prep_data_for_eval(df, metadf, umap_pca_dim,random_state=random_state)
### evaluate performance
divergence_score, entropy_score, silhouette_score = evaluate_scores(umap_codes, data, cell_type_labels,
                                                                    batch_labels, num_datasets,
                                                                    div_ent_dim, sil_dim, 'cosine', 
                                                                    random_state = random_state)  
divergence_score, entropy_score, silhouette_score
