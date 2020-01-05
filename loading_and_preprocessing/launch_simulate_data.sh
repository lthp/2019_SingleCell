#! /bin/bash

python simulate_data.py  \
        --n_samples=10 \
        --n_markers=20 \
        --n_cells_min=1000 \
        --n_cells_max=2000 \
	--n_batches=2 \
        --distribution="gamma" \
        --path_save='/Users/joannaf/Desktop/courses/DeepLearning/DL2019/project/data/simulated/toy_data_gamma_w_index.parquet' \
        --seed=234 \
	--add_ri_patient=True \
	--subset=0 \

