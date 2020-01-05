import sys
sys.path.append("..")
sys.path.append("/Users/laurieprelot/Documents/Projects/2019_Deep_learning/git_DL")
from visualisation_and_evaluation.helpers_eval_bermuda import *

path_dir = '/Users/laurieprelot/Documents/Projects/2019_Deep_learning/data/Chevrier-et-al/Outputs/output_scores/03-01-2020_18.21.02/'
fname_score = 'scores.csv'


df = extract_scores(path_dir, fname_score)
df.index_name = ['epoch']


df_best = select_best_run(df, method='div')