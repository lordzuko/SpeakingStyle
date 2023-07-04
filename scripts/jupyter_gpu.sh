#!/bin/bash

# ====================
# Options for sbatch
# ====================
#SBATCH --account=tc046
#SBATCH --job-name=style_tts
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCh --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/work/tc046/tc046/%u/work/SpeakingStyle/logs/slurm_logs/slurm-%A_output.out

# get tunneling info
XDG_RUNTIME_DIR="/work/tc046/tc046/lordzuko/"
node=$(hostname -s)
user=$(whoami)
port=8889

# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@login.cirrus.ac.uk

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# module load anaconda3/2020.11
cd /work/tc046/tc046/lordzuko/work/SpeakingStyle
source /work/tc046/tc046/lordzuko/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/tc046/tc046/lordzuko/miniconda3/lib

# Run Jupyter
conda activate fs2
jupyter-notebook --no-browser --port=${port} --ip=${node}