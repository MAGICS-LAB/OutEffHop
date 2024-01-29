#!/bin/bash
#SBATCH -A pxxxxx ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:2
#SBATCH -t 48:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 32
#SBATCH --ntasks-per-node 1 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 200G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=opt_gate ## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL
module purge
module load python-miniconda3/4.12.0
module load moose/1.0.0
module load cuda/11.4.0-gcc
module load gcc/9.2.0

conda init bash
source ~/.bashrc
#conda create -n retnet python=3.9

conda activate outlier



cd ../OutEffHop
locate
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH=${PYTHONPATH}:$(realpath "$PWD")

# Vanilla
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm.py \
--pad_to_max_length \
--wd_LN_gamma \
--with_tracking \
--report_to wandb \
--run_name test_vanilla_opt_1.3b \
--extra_tb_stats \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type opt \
--tokenizer_name facebook/opt-350m \
--max_seq_length 2048 \
--block_size 512 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 125000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 48 \
--per_device_eval_batch_size 48 \
--gradient_accumulation_steps 4 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 10000 \
--config_path model_configs/opt-12L12H.yaml \
--attn_softmax vanilla \
--output_dir output/vanilla_opt \
--resume_from_checkpoint checkpoints_path

#Softmax1
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm.py \
--pad_to_max_length \
--wd_LN_gamma \
--with_tracking \
--report_to wandb \
--run_name test_softmax1_opt_1.3b \
--extra_tb_stats \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type opt \
--tokenizer_name facebook/opt-350m \
--max_seq_length 2048 \
--block_size 512 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 125000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 48 \
--per_device_eval_batch_size 48 \
--gradient_accumulation_steps 4 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 10000 \
--config_path model_configs/opt-12L12H.yaml \
--attn_softmax softmax1 \
--output_dir output/softmax1_opt \
--resume_from_checkpoint checkpoints_path


# clipped softmax
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm.py \
--pad_to_max_length \
--wd_LN_gamma \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name test_clip_opt_1.3b \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type opt \
--tokenizer_name facebook/opt-350m \
--max_seq_length 2048 \
--block_size 512 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 125000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--gradient_accumulation_steps 8 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 10000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 10000 \
--config_path model_configs/opt-12L12H.yaml \
--alpha 12 \
--output_dir output/clip_opt \
--resume_from_checkpoint checkpoints_path


# clipped softmax1
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm.py \
--pad_to_max_length \
--wd_LN_gamma \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name revised_clip_softmax1_opt_1.3b \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type opt \
--tokenizer_name facebook/opt-350m \
--max_seq_length 2048 \
--block_size 512 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 125000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--gradient_accumulation_steps 8 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 10000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 10000 \
--config_path model_configs/opt-12L12H.yaml \
--alpha 12 \
--attn_softmax softmax1 \
--output_dir output/clip_softmax1_opt \
--resume_from_checkpoint  checkpoints_path

# gate_attention
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm.py \
--pad_to_max_length \
--wd_LN_gamma \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--run_name test_gate_opt_1.3b \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type opt \
--tokenizer_name facebook/opt-350m \
--max_seq_length 2048 \
--block_size 512 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 125000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 48 \
--per_device_eval_batch_size 48 \
--gradient_accumulation_steps 4 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 10000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 10000 \
--config_path model_configs/opt-12L12H.yaml \
--attn_gate_type conditional_per_token \
--attn_gate_init 0.25 \
--output_dir output/gate_opt \

# gate_attention softmax1
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm.py \
--pad_to_max_length \
--wd_LN_gamma \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--run_name test_gate_softmax1_opt_1.3b \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type opt \
--tokenizer_name facebook/opt-350m \
--max_seq_length 2048 \
--block_size 512 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 125000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 48 \
--per_device_eval_batch_size 48 \
--gradient_accumulation_steps 4 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 10000 \
--config_path model_configs/opt-12L12H.yaml \
--attn_gate_type conditional_per_token \
--attn_gate_init 0.25 \
--attn_softmax softmax1 \
--output_dir output/gate_softmax1_opt \
--resume_from_checkpoint checkpoint_path \