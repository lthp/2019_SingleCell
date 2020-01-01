source activate myimmuno3

logdir=/cluster/home/prelotla/GitHub/projects2019_DL_Class/bermuda_gan/logs/
script=/cluster/home/prelotla/GitHub/projects2019_DL_Class/bermuda_gan/bermuda_trial_fc.py
echo "python ${script}"|  bsub -n 3 -W 120:00 -R "rusage[mem=40000]" -o "${logdir}"

