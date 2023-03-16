#!/bin/bash

#SBATCH -o outputs/job%j.txt
#SBATCH --error errors/job%j.txt 

#SBATCH -p GPU2
#SBATCH --qos=normal
#SBATCH -J hsc_job
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu003

#SBATCH --mail-user hesicheng2001@163.com
#SBATCH --chdir /Share/UserHome/tzhao/2023/sicheng/GraduationDesign/DeepDPM

date +%c
hostname
pwd
which python

./scripts/run_experiment.sh
echo 跑完了喵