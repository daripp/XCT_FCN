#!/bin/bash

#Code by Jeffrey Neyhart
# SLURM parameters
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH -t 05:00:00   # walltime limit (HH:MM:SS) (use this for gpu-scavenger)
#SBATCH -N 1   # number of nodes
#SBATCH -n 2   # 8 processor core(s) per node X 2 threads per core
#SBATCH --mem-per-cpu 50G   # maximum memory per node
# #SBATCH -p gpu-low    # GPU node with 2 hour maximum
#SBATCH -p scavenger-gpu # GPU node with 21 day maximum, but can be killed anytime
#SBATCH --job-name="fcn_workflow_training"
#SBATCH --mail-user=user.name@email.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


##
## FCN Workflow Model Training
##

# Change working directory
cd /path/to/project/directory

# Activate the virtual environment
# 
# Note: Edit only up to, but not including, virtenv_cuda113
# 
source /path/to/virtual/environment/virtenv_cuda113/bin/activate

# Print the GPU allocation
nvidia-smi

# Run the python script
python FCN_workflow_SciNet_training_validation.py
