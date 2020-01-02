source activate myimmuno3

logdir=/cluster/home/prelotla/GitHub/projects2019_DL_Class/bermuda_gan/logs/
script=/cluster/home/prelotla/GitHub/projects2019_DL_Class/bermuda_gan/bermuda_trial_fc.py
echo "python ${script}"|  bsub -W 04:00 -R "rusage[ngpus_excl_p=1]" -o "${logdir}"

