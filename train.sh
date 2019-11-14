#!/bin/bash

#SBATCH --mem=40g
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --time=1-00
#SBATCH --priority=TOP
#SBATCH --time-min=1-00
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
module load nvidia
nvidia-smi
module load cuda/10.0
module load tensorflow

python3 train.py --exec_cap 20 --num_init_dags 10 --num_stream_dags 0 --reset_prob 5e-7 --reset_prob_min 5e-8 --reset_prob_decay 4e-10 --diff_reward_enabled 1 --num_agents 16 --model_save_interval 100 --model_folder ./models/baseline/
