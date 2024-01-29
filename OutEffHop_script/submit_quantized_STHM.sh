#!/bin/bash
#SBATCH -A pxxxxx ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:1
#SBATCH -t 24:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 24
#SBATCH --constraint=sxm
#SBATCH --ntasks-per-node 1 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 180G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=create_gate_s1
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL

module purge
module load python-miniconda3/4.12.0
module load moose/1.0.0
module load cuda/11.2.2-gcc
module load gcc/9.2.0

conda init bash
source ~/.bashrc


conda activate quantize_STHM


cd OutEffHop/OutEffHop/STanHop_outlier


############################################################### Quantize #################################################################
mkdir -p OutEffHop/OutEffHop/STanHop_outlier/results/stanhop_ETTh1_quantized

#################################### SOFTMAX ####################################################
for i in 1 2 3
do
  python quantized_main_stanhop.py  \
  --data ETTh1 \
  --in_len 168 \
  --out_len 24 \
  --seg_len 6 \
  --learning_rate 1e-4 \
  --itr 1 \
  --mode softmax \
  --use_gpu \
  --gpu 0  \
  --batch_size 128 \
  --run_name STHM_softmax \
  --e_layers 11 \
  --quantize \
  --quantize_model_path OutEffHop/OutEffHop/STanHop_outlier/checkpoints1/stanhop_ETTh1_il168_ol24_sl6_win1_fa10_dm256_nh4_el11_itr0_softmax/checkpoint.pth \
  --seed $((i * 1000)) > OutEffHop/OutEffHop/STanHop_outlier/results/stanhop_ETTh1_quantized/softmax_seq24_$i.txt
done

#################################### SOFTMAX1 ####################################################
for i in 1 2 3
do
  python quantized_main_stanhop.py  \
  --data ETTh1 \
  --in_len 168 \
  --out_len 24 \
  --seg_len 6 \
  --learning_rate 1e-4 \
  --itr 1 \
  --mode softmax1 \
  --use_gpu \
  --gpu 0  \
  --batch_size 128 \
  --run_name STHM_softmax1 \
  --e_layers 11 \
  --quantize \
  --quantize_model_path OutEffHop/OutEffHop/STanHop_outlier/checkpoints/stanhop_ETTh1_il168_ol24_sl6_win1_fa10_dm256_nh4_el11_itr0_softmax1/checkpoint.pth \
  --seed $((i * 1000)) > OutEffHop/OutEffHop/STanHop_outlier/results/stanhop_ETTh1_quantized/softmax1_seq24_$i.txt
done

# ##################################### CLIP ####################################################
for i in 1 2 3
do
  python quantized_main_stanhop.py  \
  --data ETTh1 \
  --in_len 168 \
  --out_len 24 \
  --seg_len 6 \
  --learning_rate 1e-4 \
  --itr 1 \
  --mode clip \
  --use_gpu \
  --gpu 0  \
  --batch_size 128 \
  --run_name STHM_clip \
  --e_layers 11 \
  --quantize \
  --quantize_model_path OutEffHop/OutEffHop/STanHop_outlier/checkpoints/stanhop_ETTh1_il168_ol24_sl6_win1_fa10_dm256_nh4_el11_itr0_clip/checkpoint.pth \
  --seed $((i * 1000)) > OutEffHop/OutEffHop/STanHop_outlier/results/stanhop_ETTh1_quantized/clip_seq24_$i.txt
done

##################################### CLIP Softmax1 ####################################################
for i in 1 2 3
do
  python quantized_main_stanhop.py  \
  --data ETTh1 \
  --in_len 168 \
  --out_len 24 \
  --seg_len 6 \
  --learning_rate 1e-4 \
  --itr 1 \
  --mode clip_softmax1 \
  --use_gpu \
  --gpu 0  \
  --batch_size 128 \
  --run_name STHM_clip_softmax1 \
  --e_layers 11 \
  --quantize \
  --quantize_model_path OutEffHop/OutEffHop/STanHop_outlier/checkpoints/stanhop_ETTh1_il168_ol24_sl6_win1_fa10_dm256_nh4_el11_itr0_clip_softmax1/checkpoint.pth \
  --seed $((i * 1000)) > OutEffHop/OutEffHop/STanHop_outlier/results/stanhop_ETTh1_quantized/clip_softmax1_seq24_$i.txt
done




##################################### GATED ####################################################
for i in 1 2 3
do
  python quantized_main_stanhop.py  \
  --data ETTh1 \
  --in_len 168 \
  --out_len 24 \
  --seg_len 6 \
  --learning_rate 1e-4 \
  --itr 1 \
  --mode softmax \
  --use_gpu \
  --gpu 0  \
  --batch_size 128 \
  --run_name STHM_gate \
  --e_layers 11 \
  --quantize \
  --attn_gate_type conditional_per_token \
  --attn_gate_init 0.25 \
  --quantize_model_path OutEffHop/OutEffHop/STanHop_outlier/checkpoints/stanhop_ETTh1_il168_ol24_sl6_win1_fa10_dm256_nh4_el11_itr0_gate/checkpoint.pth \
  --seed $((i * 1000)) > OutEffHop/OutEffHop/STanHop_outlier/results/stanhop_ETTh1_quantized/gate_seq24_$i.txt
done

# ##################################### GATED SOFTMAX1 ####################################################
for i in 1 2 3
do
  python quantized_main_stanhop.py  \
  --data ETTh1 \
  --in_len 168 \
  --out_len 24 \
  --seg_len 6 \
  --learning_rate 1e-4 \
  --itr 1 \
  --mode softmax1 \
  --use_gpu \
  --gpu 0  \
  --batch_size 128 \
  --run_name STHM_gate \
  --e_layers 11 \
  --quantize \
  --attn_gate_type conditional_per_token \
  --attn_gate_init 0.25 \
  --quantize_model_path OutEffHop/OutEffHop/STanHop_outlier/checkpoints/stanhop_ETTh1_il168_ol24_sl6_win1_fa10_dm256_nh4_el11_itr0_gate_softmax1/checkpoint.pth \
  --seed $((i * 1000)) > OutEffHop/OutEffHop/STanHop_outlier/results/stanhop_ETTh1_quantized/gate_softmax1_seq24_$i.txt
done

