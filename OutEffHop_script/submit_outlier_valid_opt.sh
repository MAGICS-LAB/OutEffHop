#!/bin/bash
#SBATCH -A pxxxxx ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:1
#SBATCH -t 18:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 32
#SBATCH --ntasks-per-node 1 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 200G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=OptGateS1_3## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL

module purge
module load python-miniconda3/4.12.0
module load moose/1.0.0
module load cuda/11.4.0-gcc
module load gcc/9.2.0

conda init bash
source ~/.bashrc

conda activate outlier



cd ../OutEffHop
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH=${PYTHONPATH}:$(realpath "$PWD")


#vanilla valid
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--seed 5678 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 8 \
--attn_softmax vanilla \
--data_cache_dir .hf_data  \
--model_cache_dir .hf_cache \
--model_name_or_path checkpoints_path \
--output_dir output_metrics/opt_vanilla-5678

#vanilla quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--quantize \
--quant_setup fp32_head \
--ranges_acts running_minmax \
--qmethod_acts asymmetric_uniform \
--percentile 99.999 \
--est_num_batches 4 \
--seed 6789 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 4 \
--attn_softmax vanilla \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path /output/vanilla_opt \
--output_dir  OutEffHop/output_metrics/opt_quantize_vanilla-6789uc

# softmax1
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--seed 5678 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 48 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 4 \
--attn_softmax softmax1 \
--data_cache_dir .hf_data  \
--model_cache_dir .hf_cache \
--model_name_or_path checkpoint_path \
--output_dir output_metrics/opt_softmax1-5678

#softmax1 quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--quantize \
--quant_setup fp32_head \
--ranges_acts running_minmax \
--qmethod_acts asymmetric_uniform \
--percentile 99.999 \
--est_num_batches 4 \
--seed 6789 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 48 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 4 \
--attn_softmax softmax1 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path  checkpoint_path \
--output_dir  output_metrics/opt_clipped_softmax-5678


# # clip
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--seed 5678 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 4 \
--alpha 12 \
--data_cache_dir .hf_data  \
--model_cache_dir .hf_cache \
--model_name_or_path output/clip_opt \
--output_dir output_metrics/opt_clipped_softmax-5678

# clip quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--quantize \
--quant_setup fp32_head \
--ranges_acts running_minmax \
--qmethod_acts asymmetric_uniform \
--percentile 99.999 \
--est_num_batches 4 \
--seed 6789 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 1 \
--alpha 12 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path checkpoint_path \
--output_dir output_metrics/opt_quantize_clipped_softmax-6789

# # clipped softmax1
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--seed 5678 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 4 \
--alpha 12 \
--attn_softmax softmax1 \
--data_cache_dir .hf_data  \
--model_cache_dir .hf_cache \
--model_name_or_path output/clip_softmax1_opt \
--output_dir output_metrics/opt_clipped_softmax1-5678

# clipped softmax1 quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--quantize \
--quant_setup fp32_head \
--ranges_acts running_minmax \
--qmethod_acts asymmetric_uniform \
--percentile 99.999 \
--est_num_batches 4 \
--seed 6789 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 1 \
--alpha 12 \
--attn_softmax softmax1 \
--data_cache_dir .hf_data \
--model_cache_dir .hf_cache \
--model_name_or_path output/clip_softmax1_opt \
--output_dir output_metrics/opt_quantize_clipped_softmax1-6789

# gate_attention
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--seed 5678 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 4 \
--attn_gate_type conditional_per_token \
--attn_gate_init 0.25 \
--data_cache_dir .hf_data  \
--model_cache_dir .hf_cache \
--model_name_or_path output/gate_opt \
--output_dir output_metrics/opt_gate_attention-5678

# gate_attention quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--quantize \
--quant_setup fp32_head \
--ranges_acts running_minmax \
--qmethod_acts asymmetric_uniform \
--percentile 99.999 \
--est_num_batches 4 \
--seed 6789 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 1 \
--attn_gate_type conditional_per_token \
--attn_gate_init 0.25 \
--data_cache_dir .hf_data  \
--model_cache_dir .hf_cache \
--model_name_or_path output/gate_opt \
--output_dir output_metrics/opt_quantize_gate_attention-6789

#gate_attention_softmax1
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--seed 5678 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 4 \
--attn_gate_type conditional_per_token \
--attn_gate_init 0.25 \
--attn_softmax softmax1 \
--data_cache_dir .hf_data  \
--model_cache_dir .hf_cache \
--model_name_or_path output/gate_softmax1_opt \
--output_dir output_metrics/opt_gate_attention_softmax1-5678

# gate_attention_softmax1_quantize
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
--quantize \
--quant_setup fp32_head \
--ranges_acts running_minmax \
--qmethod_acts asymmetric_uniform \
--percentile 99.999 \
--est_num_batches 4 \
--seed 6789 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 32 \
--model_type opt \
--block_size 512 \
--per_device_eval_batch_size 1 \
--attn_gate_type conditional_per_token \
--attn_gate_init 0.25 \
--attn_softmax softmax1 \
--data_cache_dir .hf_data  \
--model_cache_dir .hf_cache \
--model_name_or_path output/gate_softmax1_opt \
--output_dir output_metrics/opt_quantize_gate_attention_softmax1-6789


