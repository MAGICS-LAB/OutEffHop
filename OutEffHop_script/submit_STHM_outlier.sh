#!/bin/bash
#SBATCH -A pxxxxx ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:1
#SBATCH -t 12:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 32
#SBATCH --constraint=pcie
#SBATCH --ntasks-per-node 1 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 180G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=draw
 ## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL
module purge
module load python-miniconda3/4.12.0
module load moose/1.0.0
module load cuda/11.2.2-gcc
module load gcc/9.2.0

conda init bash
source ~/.bashrc



conda activate STHM


cd /OutEffHop/STanHop_outlier

################################################# ETTh1 ##################################################
for i in  1 2 3
do
# # # ##########################predict length 24####################################
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax1 --e_layers 11 --save_np --with_tracking 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax --e_layers 11 --save_np --with_tracking 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode clip_softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_clip_softmax1 --e_layers 11 --save_np --with_tracking 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode clip --use_gpu --gpu 0  --batch_size 128 --run_name STHM_clip --e_layers 11 --save_np --with_tracking 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_gate --e_layers 11 --attn_gate_type conditional_per_token --attn_gate_init 0.25 --with_tracking --save_np 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_gate_softmax1 --e_layers 11 --attn_gate_type conditional_per_token --attn_gate_init 0.25 --with_tracking --save_np 
done

















