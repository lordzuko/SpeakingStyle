#!/bin/bash

#SBATCH --account=tc046
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --tasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --job-name=jupyter-notebook

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