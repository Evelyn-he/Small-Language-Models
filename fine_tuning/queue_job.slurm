#!/bin/bash
#SBATCH --job-name=train_phi3
#SBATCH --output=/home/ehe/projects/def-khisti/ehe/outputs/%x-%j.out
#SBATCH --error=/home/ehe/projects/def-khisti/ehe/outputs/%x-%j.err

#SBATCH --gres=gpu:1                    # request 1 A100 GPU
#SBATCH --cpus-per-task=8           # CPU cores for preprocessing
#SBATCH --mem=64G                    # memory appropriate for smaller model
#SBATCH --time=02:00:00              # adjust time based on expected runtime

# Load necessary modules
module load gcc
module load arrow

# Activate virtual environment
source ~/gpt_env/bin/activate

# Run inference
python /home/$USER/projects/def-khisti/ehe/scripts/train.py