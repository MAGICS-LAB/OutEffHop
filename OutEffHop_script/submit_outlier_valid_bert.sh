#!/bin/bash
#SBATCH -A pxxxxx ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:2
#SBATCH -t 30:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 1 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 200G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=BertGS ## When you run squeue -u 
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



# vanilla valid
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--seed 3000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_softmax vanilla \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  checkpoint_path \
--output_dir  output_metrics/vanilla-3000

# vanilla quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--quantize \
--est_num_batches 16 \
--seed 4000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_softmax vanilla \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  /output/vanilla_bookcorpus_wiki_20000 \
--output_dir  /output_metrics/bert_quantize_vanilla-4000


# # softmax1
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--seed 3000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_softmax softmax1 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/softmax1_bookcorpus_wiki_checkpoint \
--output_dir  output_metrics/softmax1-3000

# softmax1 quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--quantize \
--est_num_batches 16 \
--seed 4000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_softmax softmax1 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path output/softmax1_bookcorpus_wiki_checkpoint \
--output_dir output_metrics/bert_quantize_softmax1-4000

# clipped softmax
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--seed 3000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_softmax "clipped(-.025:1)" \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/clipped \
--output_dir  /output_metrics/clipped_softmax-3000

# clipped softmax quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--quantize \
--est_num_batches 16 \
--seed 4000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_softmax "clipped(-.025:1)" \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/clipped \
--output_dir  /output_metrics/bert_quantize_clipped_softmax-4000

# gate_attention
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--seed 3000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_gate_type conditional_per_token \
--attn_gate_mlp \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/gate_attention_bookcorpus_wiki \
--output_dir  /output_metrics/gate_attention-3000

# gate_attention quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--quantize \
--est_num_batches 16 \
--seed 4000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_gate_type conditional_per_token \
--attn_gate_mlp \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/gate_attention_bookcorpus_wiki \
--output_dir  /output_metrics/bert_quantize_gate_attention-4000

# # clipped softmax1
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--seed 3000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_softmax "clippedsoftmax1(-.025:1)" \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/clipped_softmax1 \
--output_dir  /output_metrics/clipped_softmax1-3000



# # clipped softmax1 quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--quantize \
--est_num_batches 16 \
--seed 4000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_softmax "clippedsoftmax1(-.025:1)" \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/clipped_softmax1 \
--output_dir  output_metrics/bert_quantize_clipped_softmax1-4000

# # gate softmax1
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--seed 3000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_gate_type conditional_per_token \
--attn_gate_mlp \
--attn_softmax softmax1 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/gate_softmax1_attention_bookcorpus_wiki \
--output_dir  /output_metrics/gate_attention_softmax1-3000


# # gate softmax1 quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_mlm_config.py \
--quantize \
--est_num_batches 16 \
--seed 4000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type bert \
--max_seq_length 128 \
--mlm_probability 0.15 \
--per_device_eval_batch_size 32 \
--attn_gate_type conditional_per_token \
--attn_gate_mlp \
--attn_softmax softmax1 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  output/gate_softmax1_attention_bookcorpus_wiki \
--output_dir  /output_metrics/bert_quantize_gate_attention_softmax1-4000


