#!/bin/bash
#SBATCH -A pxxxxx ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:1
#SBATCH -t 48:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 32
#SBATCH --constraint=pcie
#SBATCH --ntasks-per-node 1 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 180G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=test_crossformer
 ## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL
module purge
module load python-miniconda3/4.12.0
module load moose/1.0.0
module load cuda/11.2.2-gcc
module load gcc/9.2.0

conda init bash
source ~/.bashrc

# conda create -n STHM python=3.8

conda activate STHM


cd /OutEffHop/STanHop_time_seeries







################################################# ETTh1 ##################################################
# mkdir -p /projects/pxxxxx/Crossformer/results/stanhop_ETTh1
for i in 1 2 3 4 5 6 7 8 9 10
do
# ##########################predict length 24####################################
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax1 --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_entmax --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 24 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_sparsemax --e_layers 11  

  ############################predict length 48####################################
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax1  --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_entmax --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax --e_layers 11  
  python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_sparsemax --e_layers 11 

  # ############################predict length 168###################################
  python main_stanhop.py  --data ETTh1  --in_len 720 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax1 --e_layers 11 
  python main_stanhop.py  --data ETTh1  --in_len 720 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_entmax --e_layers 11 
  python main_stanhop.py  --data ETTh1  --in_len 720 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax  --e_layers 11 
  python main_stanhop.py  --data ETTh1  --in_len 720 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_sparsemax --e_layers 11  

  # ############################predict length 336###################################
  python main_stanhop.py  --data ETTh1 --in_len 720 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax1  --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 720 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_entmax --e_layers 11
  python main_stanhop.py  --data ETTh1 --in_len 720 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax --e_layers 11
  python main_stanhop.py  --data ETTh1 --in_len 720 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_sparsemax --e_layers 11

  # ############################predict length 720###################################

  python main_stanhop.py  --data ETTh1 --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax1  --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_entmax  --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax  --e_layers 11 
  python main_stanhop.py  --data ETTh1 --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 128 --run_name STHM_sparsemax  --e_layers 11 
done

# ########################################## ETTm1 #############################################################
for i in 1 2 3 4 5 6 7 8 9 10
do

#     ############################predict length 24####################################
    
  python main_stanhop.py --data ETTm1 --in_len 288 --out_len 24 --seg_len 12 --learning_rate 1e-4 --itr 1 --mode softmax1  --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11 
  python main_stanhop.py --data ETTm1 --in_len 288 --out_len 24 --seg_len 12 --learning_rate 1e-4 --itr 1 --mode entmax    --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11 
  python main_stanhop.py --data ETTm1 --in_len 288 --out_len 24 --seg_len 12 --learning_rate 1e-4 --itr 1 --mode softmax   --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
  python main_stanhop.py --data ETTm1 --in_len 288 --out_len 24 --seg_len 12 --learning_rate 1e-4 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11 


    ############################predict length 48####################################
  python main_stanhop.py --data ETTm1 --in_len 288 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11 
  python main_stanhop.py --data ETTm1 --in_len 288 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11 
  python main_stanhop.py --data ETTm1 --in_len 288 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
  python main_stanhop.py --data ETTm1 --in_len 288 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11 

 ############################predict length 96####################################
    python main_stanhop.py --data ETTm1 --in_len 672 --out_len 96 --seg_len 12 --learning_rate 1e-4 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
    python main_stanhop.py --data ETTm1 --in_len 672 --out_len 96 --seg_len 12 --learning_rate 1e-4 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11
    python main_stanhop.py --data ETTm1 --in_len 672 --out_len 96 --seg_len 12 --learning_rate 1e-4 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
    python main_stanhop.py --data ETTm1 --in_len 672 --out_len 96 --seg_len 12 --learning_rate 1e-4 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11 

 ############################predict length 672####################################
    python  main_stanhop.py --data ETTm1 --in_len 672 --out_len 672 --seg_len 12 --learning_rate 1e-5 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11 
    python  main_stanhop.py --data ETTm1 --in_len 672 --out_len 672 --seg_len 12 --learning_rate 1e-5 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11 
    python  main_stanhop.py --data ETTm1 --in_len 672 --out_len 672 --seg_len 12 --learning_rate 1e-5 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
    python  main_stanhop.py --data ETTm1 --in_len 672 --out_len 672 --seg_len 12 --learning_rate 1e-5 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11 


 ############################predict length 288####################################
    python  main_stanhop.py --data ETTm1 --in_len 672 --out_len 288 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11 
    python  main_stanhop.py --data ETTm1 --in_len 672 --out_len 288 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11 
    python  main_stanhop.py --data ETTm1 --in_len 672 --out_len 288 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
    python  main_stanhop.py --data ETTm1 --in_len 672 --out_len 288 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11

done









################################################# WTH ##################################################
# ###########################predict length 48####################################
for i in 1 2 3 4 5 6 7 8 9 10
do
  ###########################predict length 24####################################
  python main_stanhop.py --data WTH --in_len 48 --out_len 24 --seg_len 4 --learning_rate 1e-4 --itr 1  --mode softmax1 --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11 
  python main_stanhop.py --data WTH --in_len 48 --out_len 24 --seg_len 4 --learning_rate 1e-4 --itr 1    --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11 
  python main_stanhop.py --data WTH --in_len 48 --out_len 24 --seg_len 4 --learning_rate 1e-4 --itr 1    --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
  python main_stanhop.py --data WTH --in_len 48 --out_len 24 --seg_len 4 --learning_rate 1e-4 --itr 1    --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11

  ##############################predict length 48####################################
  python main_stanhop.py --data WTH --in_len 48 --out_len 48 --seg_len 4 --learning_rate 1e-4 --itr 1   --mode softmax1 --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11 
  python main_stanhop.py --data WTH --in_len 48 --out_len 48 --seg_len 4 --learning_rate 1e-4 --itr 1  --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11 
  python main_stanhop.py --data WTH --in_len 48 --out_len 48 --seg_len 4 --learning_rate 1e-4 --itr 1   --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
  python main_stanhop.py --data WTH --in_len 48 --out_len 48 --seg_len 4 --learning_rate 1e-4 --itr 1   --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11 

  ############################predict length 168###################################

  python main_stanhop.py --data WTH --in_len 336 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1  --mode softmax1 --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11
  python main_stanhop.py --data WTH --in_len 336 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1  --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
  python main_stanhop.py --data WTH --in_len 336 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1  --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11
  python main_stanhop.py --data WTH --in_len 336 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1  --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11 

  ############################predict length 720###################################
  python  main_stanhop.py --data WTH --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 1    --mode softmax1 --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11 
  python  main_stanhop.py --data WTH --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 1    --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
  python  main_stanhop.py --data WTH --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 1    --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11 
  python  main_stanhop.py --data WTH --in_len 720 --out_len 720 --seg_len 24 --learning_rate 1e-5 --itr 1    --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11 

  python  main_stanhop.py --data WTH --in_len 336 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode sparsemax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_sparsemax --e_layers 11 
  python  main_stanhop.py --data WTH --in_len 336 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax --e_layers 11 
  python  main_stanhop.py --data WTH --in_len 336 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 64 --run_name STHM_softmax1 --e_layers 11 
  python  main_stanhop.py --data WTH --in_len 336 --out_len 336 --seg_len 24 --learning_rate 1e-5 --itr 1 --mode entmax --use_gpu --gpu 0  --batch_size 64 --run_name STHM_entmax --e_layers 11 

done