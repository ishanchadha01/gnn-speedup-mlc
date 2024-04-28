#!/bin/bash
#SBATCH -J mlc-gnn-preprocess                                                # Job name
#SBATCH --account=gts-ychen3538                                 # Tracking account
#SBATCH -N 8 --ntasks-per-node 8                             # Number of nodes and cores per node required
#SBATCH -t720                                                   # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                               # QOS Name
#SBATCH -o preprocess_mlc_gnn_10m.out                                   # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                              # Mail preferences
#SBATCH --mail-user=ichadha3@gatech.edu                         # E-mail address for notifications

cd /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/mlc/gnn-speedup-mlc/data_pre

module load anaconda3
module load gcc

conda activate mlc
python --version
python process_data.py