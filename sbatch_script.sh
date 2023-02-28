#!/bin/bash

#SBATCH -o outputs/job%j.txt
#SBATCH --error errors/job%j.txt 

#SBATCH -p GPU2
#SBATCH --qos=normal
#SBATCH -J deepDPM
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:8


#SBATCH --mail-user hesicheng2001@163.com  

cd ~/2023/sicheng/DeepDPM
pwd
date +%c
./run_experiment.sh
echo 跑完了喵