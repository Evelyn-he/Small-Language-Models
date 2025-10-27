#!/bin/bash
#SBATCH --job-name=gptoss_api
#SBATCH --account=def-yourprof          # replace with your actual project allocation name
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=72:00:00

module load python/3.11
source venv/bin/activate    # since your venv is inside this folder

cd /project/def-yourprof/$USER/Project2025206/Small-Language-Models
srun python llm_server.py
