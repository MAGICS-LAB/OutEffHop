#!/bin/bash
#SBATCH -A pxxxxx ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:2
#SBATCH -t 48:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --constraint=sxm
#SBATCH --cpus-per-task 24 
#SBATCH --ntasks-per-node 2
#SBATCH --mem 180G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=ViT-gate_softmax1  ## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL

module purge
module load python-miniconda3/4.12.0
module load moose/1.0.0
module load cuda/11.4.0-gcc
module load gcc/9.2.0
module load git
module load nvidia-apex/23.08

conda init bash
source ~/.bashrc
#conda create -n retnet python=3.9

conda activate outlier



cd OutEffHop


# ############# clipsoftmax1 #########################################################################
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 run_vit.py \
    --data-dir path_imagenet-1k-0.1 \
    --model vit_small_patch16_224 \
    --amp \
    --opt adamw \
    --weight-decay 0.03 \
    --lr 0.001 \
    --min-lr 0.00001 \
    --warmup-lr 0.000001 \
    --warmup-epochs 10 \
    --decay-epochs 15 \
    --decay-rate 0.1 \
    --aa rand-m9-mstd0.5 \
    --mixup 0.2 \
    --cutmix 0.2 \
    --smoothing 0.1 \
    --color-jitter 0.4 \
    --batch-size 512 \
    --with_tracking \
    --report_to wandb \
    --seed 1000 \
    --grad-accum-steps 5 \
    --tb_scalar_log_interval 100 \
    --validation-batch-size 512\
    --epochs 150 \
    --checkpoint-hist 5 \
    --attn_softmax 'clippedsoftmax1(-.00001:1)' \
    --gamma -0.0001 \
    --output output/clip_softmax1 _imagenet_0.1 \
    --run_name pre_norm_clip_softmax1_ViT-s_16_Imagenet-1k_0.1 \




# ############### clip #########################################################################
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 run_vit.py \
    --data-dir path_imagenet-1k-0.1 \
    --model vit_small_patch16_224 \
    --amp \
    --opt adamw \
    --weight-decay 0.03 \
    --lr 0.001 \
    --min-lr 0.00001 \
    --warmup-lr 0.000001 \
    --warmup-epochs 10 \
    --decay-epochs 15 \
    --decay-rate 0.1 \
    --aa rand-m9-mstd0.5 \
    --mixup 0.2 \
    --cutmix 0.2 \
    --smoothing 0.1 \
    --color-jitter 0.4 \
    --batch-size 512 \
    --with_tracking \
    --report_to wandb \
    --seed 1000 \
    --grad-accum-steps 5 \
    --tb_scalar_log_interval 100 \
    --validation-batch-size 512\
    --epochs 150 \
    --checkpoint-hist 5 \
    --attn_softmax 'clipped(-.00001:1)' \
    --gamma -0.0001 \
    --output output/clip_imagenet_0.1 \
    --run_name pre_norm_clip_ViT-s_16_Imagenet-1k_0.1 \



# ############## vanilla #########################################################################
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 run_vit.py \
    --data-dir path_imagenet-1k-0.1 \
    --model vit_small_patch16_224 \
    --amp \
    --opt adamw \
    --weight-decay 0.03 \
    --lr 0.001 \
    --min-lr 0.00001 \
    --warmup-lr 0.000001 \
    --warmup-epochs 10 \
    --decay-epochs 15 \
    --decay-rate 0.1 \
    --aa rand-m9-mstd0.5 \
    --mixup 0.2 \
    --cutmix 0.2 \
    --smoothing 0.1 \
    --color-jitter 0.4 \
    --batch-size 512 \
    --with_tracking \
    --report_to wandb \
    --seed 1000 \
    --grad-accum-steps 5 \
    --tb_scalar_log_interval 100 \
    --validation-batch-size 512\
    --epochs 150 \
    --checkpoint-hist 5 \
    --attn_softmax vanilla \
    --output output/vanilla_imagenet_0.1 \
    --run_name pre_norm_vanilla_ViT-s_16_Imagenet-1k_0.1 \




# ############## softmax1 #########################################################################
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 run_vit.py \
    --data-dir path_imagenet-1k-0.1 \
    --model vit_small_patch16_224 \
    --amp \
    --opt adamw \
    --weight-decay 0.03 \
    --lr 0.001 \
    --min-lr 0.00001 \
    --warmup-lr 0.000001 \
    --warmup-epochs 10 \
    --decay-epochs 15 \
    --decay-rate 0.1 \
    --aa rand-m9-mstd0.5 \
    --mixup 0.2 \
    --cutmix 0.2 \
    --smoothing 0.1 \
    --color-jitter 0.4 \
    --batch-size 512 \
    --with_tracking \
    --report_to wandb \
    --seed 1000 \
    --grad-accum-steps 5 \
    --tb_scalar_log_interval 100 \
    --validation-batch-size 512\
    --epochs 150 \
    --checkpoint-hist 5 \
    --attn_softmax softmax1 \
    --output output/softmax1_imagenet_0.1 \
    --run_name pre_norm_softmax1_ViT-s_16_Imagenet-1k_0.1 \





# ############## gate #########################################################################
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 run_vit.py \
    --data-dir path_imagenet-1k-0.1 \
    --model vit_small_patch16_224 \
    --amp \
    --opt adamw \
    --weight-decay 0.03 \
    --lr 0.001 \
    --min-lr 0.00001 \
    --warmup-lr 0.000001 \
    --warmup-epochs 10 \
    --decay-epochs 15 \
    --decay-rate 0.1 \
    --aa rand-m9-mstd0.5 \
    --mixup 0.2 \
    --cutmix 0.2 \
    --smoothing 0.1 \
    --color-jitter 0.4 \
    --batch-size 512 \
    --with_tracking \
    --report_to wandb \
    --seed 1000 \
    --grad-accum-steps 5 \
    --tb_scalar_log_interval 100 \
    --validation-batch-size 512\
    --epochs 150 \
    --checkpoint-hist 5 \
    --attn_gate_type conditional_per_token \
    --attn_gate_init 0.25 \
    --output output/gate_imagenet \
    --attn_softmax vanilla \
    --run_name pre_norm_gate_ViT-s_16_Imagenet-1k_0.1 \



####################################################### Gate softmax1 #########################################################################
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 run_vit.py \
    --data-dir path_imagenet-1k-0.1 \
    --model vit_small_patch16_224 \
    --amp \
    --opt adamw \
    --weight-decay 0.03 \
    --lr 0.001 \
    --min-lr 0.00001 \
    --warmup-lr 0.000001 \
    --warmup-epochs 10 \
    --decay-epochs 15 \
    --decay-rate 0.1 \
    --aa rand-m9-mstd0.5 \
    --mixup 0.2 \
    --cutmix 0.2 \
    --smoothing 0.1 \
    --color-jitter 0.4 \
    --batch-size 512 \
    --with_tracking \
    --report_to wandb \
    --seed 1000 \
    --grad-accum-steps 5 \
    --tb_scalar_log_interval 100 \
    --validation-batch-size 512\
    --epochs 150 \
    --checkpoint-hist 5 \
    --attn_gate_type conditional_per_token \
    --attn_gate_init 0.25 \
    --output output/gate_softmax1_imagenet \
    --attn_softmax softmax1 \
    --run_name pre_norm_gate_softmax1_ViT-s_16_Imagenet-1k_0.1 \



