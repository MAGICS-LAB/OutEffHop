#!/bin/bash
#SBATCH -A pxxxxx ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:2
#SBATCH -t 48:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 32
#SBATCH --ntasks-per-node 1 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 200G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=bert_new200000 ## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL

module purge
module load python-miniconda3/4.12.0
module load moose/1.0.0
module load cuda/11.4.0-gcc
module load gcc/9.2.0

conda init bash
source ~/.bashrc

conda activate outlier





cd OutEffHop
locate
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH=${PYTHONPATH}:$(realpath "$PWD")

# Vanilla
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_mlm_origin.py \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name new_vanilla_bookcorpus_wiki_200000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type bert \
--tokenizer_name bert-base-uncased \
--max_seq_length 128 \
--mlm_probability 0.15 \
--learning_rate 0.0001 \
--lr_scheduler_type linear \
--max_train_steps 200000 \
--num_warmup_steps 10000 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--config_name bert-base-uncased \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 100000 \
--attn_softmax vanilla \
--output_dir output/vanilla_bookcorpus_wiki_20000 \
# --resume_from_checkpoint /output/vanilla/checkpoints/checkpoint_39

# Vanilla checkpoint
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_mlm_origin.py \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name bert_vanilla_resume \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type bert \
--tokenizer_name bert-base-uncased \
--max_seq_length 128 \
--mlm_probability 0.15 \
--learning_rate 0.0001 \
--lr_scheduler_type linear \
--max_train_steps 1000000 \
--num_warmup_steps 10000 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--config_name bert-base-uncased \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 100000 \
--attn_softmax vanilla \
--output_dir output/vanilla_bookcorpus_wiki \
--resume_from_checkpoint output/vanilla_bookcorpus_wiki/checkpoints/checkpoint_39


# Softmax1

accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_mlm_origin.py \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name checkpoint_test_softmax1_bookcorpus_wiki \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type bert \
--tokenizer_name bert-base-uncased \
--max_seq_length 128 \
--mlm_probability 0.15 \
--learning_rate 0.0001 \
--lr_scheduler_type linear \
--max_train_steps 200000 \
--num_warmup_steps 10000 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--config_name bert-base-uncased \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 100000 \
--attn_softmax softmax1 \
--output_dir output/softmax1_bookcorpus_wiki_checkpoint



# clipped softmax
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_mlm_origin.py \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name test_origin_launch_job_clip_bookcorpus_wiki \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type bert \
--tokenizer_name bert-base-uncased \
--max_seq_length 128 \
--mlm_probability 0.15 \
--learning_rate 0.0001 \
--lr_scheduler_type linear \
--max_train_steps 200000 \
--num_warmup_steps 10000 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--config_name bert-base-uncased \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 100000 \
--attn_softmax "clipped(-.025:1)" \
--output_dir output/clipped \


clipped softmax1
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_mlm_origin.py \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name test_origin_launch_job_clip_softmax1_bookcorpus_wiki \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type bert \
--tokenizer_name bert-base-uncased \
--max_seq_length 128 \
--mlm_probability 0.15 \
--learning_rate 0.0001 \
--lr_scheduler_type linear \
--max_train_steps 200000 \
--num_warmup_steps 10000 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--config_name bert-base-uncased \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 100000 \
--attn_softmax "clippedsoftmax1(-.025:1)" \
--output_dir output/clipped_softmax1 \

# gate_attention
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_mlm_origin.py \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name test_origin_launch_job_gate_attention_bookcorpus_wiki \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type bert \
--tokenizer_name bert-base-uncased \
--max_seq_length 128 \
--mlm_probability 0.15 \
--learning_rate 0.0001 \
--lr_scheduler_type linear \
--max_train_steps 200000 \
--num_warmup_steps 10000 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--config_name bert-base-uncased \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 100000 \
--attn_gate_type conditional_per_token \
--attn_gate_mlp \
--output_dir output/gate_attention_bookcorpus_wiki \


# gate_attention softmax1
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_mlm_origin.py \
--with_tracking \
--report_to wandb \
--extra_tb_stats \
--seed 1000 \
--run_name test_origin_launch_job_gate_softmax1_attention_bookcorpus_wiki \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_type bert \
--tokenizer_name bert-base-uncased \
--max_seq_length 128 \
--mlm_probability 0.15 \
--learning_rate 0.0001 \
--lr_scheduler_type linear \
--max_train_steps 200000 \
--num_warmup_steps 10000 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--config_name bert-base-uncased \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 100000 \
--attn_gate_type conditional_per_token \
--attn_gate_mlp \
--attn_softmax softmax1 \
--output_dir output/gate_softmax1_attention_bookcorpus_wiki \