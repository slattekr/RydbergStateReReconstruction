#!/usr/bin/env bash
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:1  
#SBATCH --output=outputs/slurm-%A_%a.out 
#SBATCH --mem=10000
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --mail-user=msmoss@uwaterloo.ca
#SBATCH --mail-type=ALL

module load python/3

source ReconstructRydberg/bin/activate

python twoD_VMC.py

