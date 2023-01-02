#!/usr/bin/env bash
#SBATCH -t 3-00:00:00
#SBATCH --gpus-per-node=p100  
#SBATCH --output=outputs/slurm-%A_%a.out 
#SBATCH --mem=20000
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --mail-user=msmoss@uwaterloo.ca
#SBATCH --mail-type=ALL

module load cuda/11.2.2 cudnn/8.2.0

nvidia-smi

export TF_GPU_ALLOCATOR=cuda_malloc_async

source ReconstructRydberg/bin/activate

python 12.455_vmc.py && python 12.955_vmc.py && python 13.455_vmc.py 

