#!/bin/bash
#SBATCH --job-name=gptoss_api
#SBATCH --account=def-yourprof          # your project allocation
#SBATCH --nodes=2                        # number of nodes
#SBATCH --gres=gpu:4                     # 4 GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=72:00:00

module load python/3.11
source venv/bin/activate

cd ~/Project2025206/Small-Language-Models
srun python llm_server.py
