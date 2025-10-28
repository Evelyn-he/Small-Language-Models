#!/bin/bash
#SBATCH --job-name=gptoss_api
#SBATCH --account=def-khisti
#SBATCH --partition=gpu                # 👈 Add this line
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                   # 👈 Will now be recognized
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=llm_server.out
#SBATCH --error=llm_server.err

module load python/3.11
source ~/projects/def-khisti/Project2025206/Small-Language-Models/venv/bin/activate

cd ~/projects/def-khisti/Project2025206/Small-Language-Models
srun python llm_server.py
